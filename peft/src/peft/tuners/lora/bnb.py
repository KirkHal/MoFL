
import warnings
import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils.other import transpose
from itertools import chain
from .layer import LoraLayer
from .utils import Router
from .utils import rank0_print


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
                self,
                adapter_name,
                in_features,
                out_features,
                r: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0.0,
                **kwargs,
        ) -> None:
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.set_adapter(adapter_name)

        def merge(self):
            if self.merged:
                warnings.warn(
                    f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                    f"You are now additionally merging {','.join(self.active_adapters)}."
                )
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                warnings.warn(
                    "Merge lora module to 8-bit linear may get different generations due to rounding errors."
                )
                lora_data = self.get_delta_weight(active_adapter)

                if self.state.SCB is None:
                    self.state.SCB = self.weight.SCB
                # Dequantize the result of identity matrix and int8 weight because bitsandbytes does not support int8
                # dequantization directly
                im = torch.eye(self.weight.data.shape[-1]).contiguous().half().to(self.weight.device)
                im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
                im, Sim = bnb.functional.transform(im, "col32")

                if self.state.CxB is None:
                    self.state.CxB, self.state.SB = bnb.functional.transform(
                        self.weight.data, to_order=self.state.formatB
                    )
                out32, Sout32 = bnb.functional.igemmlt(im, self.state.CxB, Sim, self.state.SB)
                output = bnb.functional.mm_dequant(out32, Sout32, SCim, self.state.SCB, bias=None).t()
                w_data = output.to(lora_data.dtype).to(lora_data.device) + lora_data
                self.weight = bnb.nn.Int8Params(
                    w_data.to("cpu"), requires_grad=False, has_fp16_weights=self.weight.has_fp16_weights
                ).to(self.weight.device)
                self.state.reset_grads()
                self.merged_adapters.append(active_adapter)
                self.merged = True

        def unmerge(self):
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return
            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.lora_A.keys():
                    continue
                warnings.warn(
                    "Unmerge lora module to 8-bit linear may get different generations due to rounding errors."
                )
                lora_data = self.get_delta_weight(active_adapter)

                if self.state.SCB is None:
                    self.state.SCB = self.weight.SCB
                im = torch.eye(self.weight.data.shape[-1]).contiguous().half().to(self.weight.device)
                im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
                im, Sim = bnb.functional.transform(im, "col32")

                if self.state.CxB is None:
                    self.state.CxB, self.state.SB = bnb.functional.transform(
                        self.weight.data, to_order=self.state.formatB
                    )
                out32, Sout32 = bnb.functional.igemmlt(im, self.state.CxB, Sim, self.state.SB)
                output = bnb.functional.mm_dequant(out32, Sout32, SCim, self.state.SCB, bias=None).t()
                w_data = output.to(lora_data.dtype).to(lora_data.device) - lora_data
                self.weight = bnb.nn.Int8Params(
                    w_data.to("cpu"), requires_grad=False, has_fp16_weights=self.weight.has_fp16_weights
                ).to(self.weight.device)
                self.state.reset_grads()
                self.merged = False

        def get_delta_weight(self, adapter):
            return (
                    transpose(
                        self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
                        False,
                    )
                    * self.scaling[adapter]
            )

        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = super().forward(x)
            elif self.merged:
                result = super().forward(x)
            else:
                result = super().forward(x)
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.lora_A.keys():
                        continue
                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        compute_dtype = lora_A.weight.dtype
                        if x.dtype != compute_dtype:
                            x = x.to(compute_dtype)
                    output = lora_B(lora_A(dropout(x)))
                    if requires_conversion:
                        output = output.to(expected_dtype)
                    output = output * scaling
                    result += output

            return result

if is_bnb_4bit_available():

    class Linear4bit(bnb.nn.Linear4bit, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
                self,
                adapter_name,
                in_features,
                out_features,
                r: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0.0,
                **kwargs,
        ) -> None:
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                compute_dtype=kwargs.get("compute_dtype", torch.float32),
                compress_statistics=kwargs.get("compress_statistics", True),
                quant_type=kwargs.get("quant_type", "nf4"),
            )
            LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.lora_A_Net = nn.ModuleDict({})
            self.lora_B_Net = nn.ModuleDict({})
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)


        def merge(self):
            if self.merged:
                warnings.warn(
                    f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                    f"You are now additionally merging {','.join(self.active_adapters)}."
                )
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                warnings.warn(
                    "Merge lora module to 4-bit linear may get different generations due to rounding errors."
                )
                # Refer to https://gist.github.com/ChrisHayduk/1a53463331f52dca205e55982baf9930
                kwargs = self.weight.__dict__
                lora_data = self.get_delta_weight(active_adapter)
                w_data = bnb.functional.dequantize_4bit(self.weight.data, self.weight.quant_state) + lora_data
                self.weight = bnb.nn.Params4bit(w_data.to("cpu"), requires_grad=False, **kwargs).to(self.weight.device)
                self.merged = True

        def unmerge(self):
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return
            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.lora_A.keys():
                    continue
                warnings.warn(
                    "Unmerge lora module to 4-bit linear may get different generations due to rounding errors."
                )
                kwargs = self.weight.__dict__
                lora_data = self.get_delta_weight(active_adapter)
                w_data = bnb.functional.dequantize_4bit(self.weight.data, self.weight.quant_state) - lora_data
                self.weight = bnb.nn.Params4bit(w_data.to("cpu"), requires_grad=False, **kwargs).to(self.weight.device)
                self.merged = False

        def get_delta_weight(self, adapter):
            return (
                    transpose(
                        self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
                        False,
                    )
                    * self.scaling[adapter]
            )

        def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
            self.r[adapter_name] = r
            self.lora_alpha[adapter_name] = lora_alpha
            self.scaling[adapter_name] = lora_alpha / r
            if lora_dropout > 0.0:
                lora_dropout_layer = nn.Dropout(p=lora_dropout)
            else:
                lora_dropout_layer = nn.Identity()

            self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
            # Actual trainable parameters
            if r > 0:
                self.lora_A_Net["A"] = nn.Linear(self.in_features, r, bias=False)
                self.lora_B_Net["B"] = nn.Linear(r, self.out_features, bias=False)

                self.lora_A_Net["A"].weight.data = self.lora_A_Net["A"].weight.data.to(torch.bfloat16)
                self.lora_B_Net["B"].weight.data = self.lora_B_Net["B"].weight.data.to(torch.bfloat16)

            if init_lora_weights:
                self.reset_lora_parameters(adapter_name)

            weight = getattr(self, "weight", None)
            if weight is not None:
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)

        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            result = super().forward(x)
            result = result.clone()

            scaling = self.scaling["expert-0"]
            dropout = self.lora_dropout["expert-0"]

            self.lora_A_Net["A"].weight.data = self.lora_A_Net["A"].weight.data.to(torch.bfloat16)
            self.lora_B_Net["B"].weight.data = self.lora_B_Net["B"].weight.data.to(torch.bfloat16)

            lora_output = self.lora_B_Net["B"](self.lora_A_Net["A"](dropout(x))) * scaling

            return result + lora_output



    class MoELinear4bit(bnb.nn.Linear4bit, LoraLayer):
        def __init__(
                self,
                adapter_name,
                in_features,
                out_features,
                r: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0.0,
                expert_nums=None,
                **kwargs,
        ) -> None:
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                compute_dtype=kwargs.get("compute_dtype", torch.float32),
                compress_statistics=kwargs.get("compress_statistics", True),
                quant_type=kwargs.get("quant_type", "nf4"),
            )
            LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.experts = None
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

            # self.router = Router(self.weight.shape[0], 4, expert_nums).to(self.weight.device)  # 70B
            self.router = Router(self.weight.shape[-1], 128, expert_nums).to(self.weight.device)  # 7B

            rank0_print("MoELinear4bit")

        def _gating(self, x, ):
            weights = self.router(x)  # [batch_size, seq_len, experts_num]
            weights = F.softmax(weights, dim=-1).unsqueeze(dim=-2)  # [batch_size, seq_len, 1, experts_num]

            return weights

        def merge(self):
            raise NotImplementedError

        def unmerge(self):
            raise NotImplementedError

        def get_delta_weight(self, adapter):
            raise NotImplementedError

        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            # 1 router
            gating_distribution = self._gating(x, )

            result = super().forward(x)
            result = result.clone()

            for active_adapter in self.experts[:1]:

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = x.to(lora_A.weight.dtype)

                output = lora_B(lora_A(dropout(x))) * scaling  # [batch_size, seq_len, emb_dim]

            output = output.unsqueeze(dim=-2)  # [batch_size, seq_len, 1, emb_dim]

            for active_adapter in self.experts[1:]:

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = x.to(lora_A.weight.dtype)

                tmp = lora_B(lora_A(dropout(x)))
                if requires_conversion:
                    tmp = tmp.to(expected_dtype)
                tmp = (tmp * scaling).unsqueeze(dim=-2)
                output = torch.cat((output, tmp), dim=-2)

            result += torch.matmul(gating_distribution, output.to(gating_distribution.dtype)).squeeze(dim=-2)



    class MoELinear4bitV2(bnb.nn.Linear4bit, LoraLayer):
        """ Multi-B module """
        def __init__(
                self,
                adapter_name,
                in_features,
                out_features,
                r: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0.0,
                expert_nums=None,
                **kwargs,
        ) -> None:
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                compute_dtype=kwargs.get("compute_dtype", torch.float32),
                compress_statistics=kwargs.get("compress_statistics", True),
                quant_type=kwargs.get("quant_type", "nf4"),
            )
            LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)

            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, expert_nums)

            # self.router = Router(self.weight.shape[0], expert_nums).to(self.weight.device)  # 70B
            self.router = Router(self.weight.shape[-1], expert_nums).to(self.weight.device)  # 7B
            for p in self.router.parameters():
                p.data = p.data.to("cuda")
            # rank0_print("Adding MoELinear4bitV2".center(200, "<"))

        def _gating(self, x, ):
            weights = self.router(x)  # [batch_size, seq_len, experts_num]
            weights = weights.unsqueeze(dim=-2)  # [batch_size, seq_len, 1, experts_num]

            return weights

        def merge(self):
            raise NotImplementedError

        def unmerge(self):
            raise NotImplementedError

        def get_delta_weight(self, adapter):
            raise NotImplementedError

        def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, expert_nums):
            if r <= 0:
                raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
            self.r[adapter_name] = r
            self.lora_alpha[adapter_name] = lora_alpha
            if lora_dropout > 0.0:
                lora_dropout_layer = nn.Dropout(p=lora_dropout)
            else:
                lora_dropout_layer = nn.Identity()

            self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
            # Actual trainable parameters
            if r > 0:
                self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
                for i in range(expert_nums):
                    self.lora_B[f"expert-{i}"] = nn.Linear(r, self.out_features, bias=False)
                self.scaling[adapter_name] = lora_alpha / r
            if init_lora_weights:
                self.reset_lora_parameters(adapter_name)

            weight = getattr(self, "weight", None)
            if weight is not None:
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)

        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            gating_distribution = self._gating(x)
            result = super().forward(x).clone()

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(self.lora_A["expert-0"].weight.dtype)

            # Apply the first adapter
            lora_A = self.lora_A["expert-0"]
            dropout = self.lora_dropout["expert-0"]
            scaling = self.scaling["expert-0"]
            outputs = [lora_B(lora_A(dropout(x))) * scaling for active_adapter, lora_B in list(self.lora_B.items())[:1]]
            outputs = [output.unsqueeze(dim=-2) for output in outputs]

            # Apply the rest of the adapters
            for active_adapter, lora_B in list(self.lora_B.items())[1:]:
                tmp = lora_B(lora_A(dropout(x))) * scaling
                if requires_conversion:
                    tmp = tmp.to(expected_dtype)
                outputs.append(tmp.unsqueeze(dim=-2))

            output = torch.cat(outputs, dim=-2)

            result += torch.matmul(gating_distribution, output.to(gating_distribution.dtype)).squeeze(dim=-2)

            return result



    class MoELinear4bitV3(bnb.nn.Linear4bit, LoraLayer):
        """ Multi-A module """
        def __init__(
                self,
                adapter_name,
                in_features,
                out_features,
                r: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0.0,
                expert_nums=None,
                **kwargs,
        ) -> None:
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                compute_dtype=kwargs.get("compute_dtype", torch.float32),
                compress_statistics=kwargs.get("compress_statistics", True),
                quant_type=kwargs.get("quant_type", "nf4"),
            )
            LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)

            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, expert_nums)

            # self.router = Router(self.weight.shape[0], expert_nums).to(self.weight.device)  # 70B
            self.router = Router(self.weight.shape[-1], expert_nums).to(self.weight.device)  # 7B
            for p in self.router.parameters():
                p.data = p.data.to("cuda")
            # rank0_print("Adding MoELinear4bitV3".center(200, "<"))

        def _gating(self, x, ):
            weights = self.router(x)  # [batch_size, seq_len, experts_num]
            weights = weights.unsqueeze(dim=-2)  # [batch_size, seq_len, 1, experts_num]

            return weights

        def merge(self):
            raise NotImplementedError

        def unmerge(self):
            raise NotImplementedError

        def get_delta_weight(self, adapter):
            raise NotImplementedError

        def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, expert_nums):
            if r <= 0:
                raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
            self.r[adapter_name] = r
            self.lora_alpha[adapter_name] = lora_alpha
            if lora_dropout > 0.0:
                lora_dropout_layer = nn.Dropout(p=lora_dropout)
            else:
                lora_dropout_layer = nn.Identity()

            self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
            # Actual trainable parameters
            if r > 0:
                for i in range(expert_nums):
                    self.lora_A[f"expert-{i}"] = nn.Linear(self.in_features, r, bias=False)

                self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
                self.scaling[adapter_name] = lora_alpha / r
            if init_lora_weights:
                self.reset_lora_parameters(adapter_name)

            weight = getattr(self, "weight", None)
            if weight is not None:
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)

        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            gating_distribution = self._gating(x)
            result = super().forward(x).clone()

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(self.lora_A["expert-0"].weight.dtype)

            # Apply the first adapter
            lora_B = self.lora_B["expert-0"]
            dropout = self.lora_dropout["expert-0"]
            scaling = self.scaling["expert-0"]
            outputs = [lora_B(lora_A(dropout(x))) * scaling for active_adapter, lora_A in list(self.lora_A.items())[:1]]
            outputs = [output.unsqueeze(dim=-2) for output in outputs]

            # Apply the rest of the adapters
            for active_adapter, lora_A in list(self.lora_A.items())[1:]:
                tmp = lora_B(lora_A(dropout(x))) * scaling
                if requires_conversion:
                    tmp = tmp.to(expected_dtype)
                outputs.append(tmp.unsqueeze(dim=-2))

            output = torch.cat(outputs, dim=-2)
            result += torch.matmul(gating_distribution, output.to(gating_distribution.dtype)).squeeze(dim=-2)

            return result



    class MoELinear4bitV4(bnb.nn.Linear4bit, LoraLayer):
        """ 1 * Con module + 1 * Div module """
        def __init__(
                self,
                adapter_name,
                in_features,
                out_features,
                r: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0.0,
                expert_nums=None,
                **kwargs,
        ) -> None:
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                compute_dtype=kwargs.get("compute_dtype", torch.float32),
                compress_statistics=kwargs.get("compress_statistics", True),
                quant_type=kwargs.get("quant_type", "nf4"),
            )
            LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)

            self.lora_A_Net1 = nn.ModuleDict({})
            self.lora_B_Net1 = nn.ModuleDict({})
            self.lora_A_Net2 = nn.ModuleDict({})
            self.lora_B_Net2 = nn.ModuleDict({})

            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, expert_nums)

            self.router_1 = Router(self.weight.shape[-1], expert_nums).to(self.weight.device)
            self.router_2 = Router(self.weight.shape[-1], expert_nums).to(self.weight.device)
            self.SoftRouter = Router(self.weight.shape[-1], 2).to(self.weight.device)


            for p in chain(self.router_1.parameters(), self.router_2.parameters()):
                p.data = p.data.to("cuda")
            for p in self.SoftRouter.parameters():
                p.data = p.data.to("cuda")

        def _gating(self, x, ):
            weights = []
            for router in (self.router_1, self.router_2):
                weights.append(router(x).unsqueeze(dim=-2))  # [batch_size, seq_len, 1, experts_num]
            return weights

        def merge(self):
            raise NotImplementedError

        def unmerge(self):
            raise NotImplementedError

        def get_delta_weight(self, adapter):
            raise NotImplementedError

        def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, expert_nums):
            if r <= 0:
                raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
            self.r[adapter_name] = r
            self.lora_alpha[adapter_name] = lora_alpha
            if lora_dropout > 0.0:
                lora_dropout_layer = nn.Dropout(p=lora_dropout)
            else:
                lora_dropout_layer = nn.Identity()

            self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
            # Actual trainable parameters
            if r > 0:
                self.lora_A_Net1[adapter_name] = nn.Linear(self.in_features, r, bias=False)
                for i in range(expert_nums):
                    self.lora_B_Net1[f"expert-{i}"] = nn.Linear(r, self.out_features, bias=False)

                for i in range(expert_nums):
                    self.lora_A_Net2[f"expert-{i}"] = nn.Linear(self.in_features, r, bias=False)
                self.lora_B_Net2[adapter_name] = nn.Linear(r, self.out_features, bias=False)
                self.scaling[adapter_name] = lora_alpha / r
            if init_lora_weights:
                self.reset_lora_parameters(adapter_name)

            weight = getattr(self, "weight", None)
            if weight is not None:
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)

        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            gating_distribution_1, gating_distribution_2 = self._gating(x)
            # rank0_print(f"gating_distribution_2: {gating_distribution_2}", gating_distribution_2.shape)

            result = super().forward(x).clone()
            # rank0_print(f"result: {result}", result.shape)

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(self.lora_A_Net1["expert-0"].weight.dtype)

            # Process Network_1
            lora_A = self.lora_A_Net1["expert-0"]
            dropout = self.lora_dropout["expert-0"]
            scaling = self.scaling["expert-0"]
            outputs_1 = [lora_B(lora_A(dropout(x))) * scaling for active_adapter, lora_B in
                         list(self.lora_B_Net1.items())[:1]]
            outputs_1 = [output.unsqueeze(dim=-2) for output in outputs_1]
            for active_adapter, lora_B in list(self.lora_B_Net1.items())[1:]:
                tmp = lora_B(lora_A(dropout(x))) * scaling
                if requires_conversion:
                    tmp = tmp.to(expected_dtype)
                outputs_1.append(tmp.unsqueeze(dim=-2))
            output_1 = torch.cat(outputs_1, dim=-2)
            output_1 = torch.matmul(gating_distribution_1, output_1.to(gating_distribution_1.dtype)).squeeze(dim=-2)
            # rank0_print(f"output_1: {output_1}", output_1.shape)
            result += output_1

            # Process Network_2
            lora_B = self.lora_B_Net2["expert-0"]
            dropout = self.lora_dropout["expert-0"]
            scaling = self.scaling["expert-0"]
            outputs_2 = [lora_B(lora_A(dropout(x))) * scaling for active_adapter, lora_A in
                         list(self.lora_A_Net2.items())[:1]]
            outputs_2 = [output.unsqueeze(dim=-2) for output in outputs_2]
            for active_adapter, lora_A in list(self.lora_A_Net2.items())[1:]:
                tmp = lora_B(lora_A(dropout(x))) * scaling
                if requires_conversion:
                    tmp = tmp.to(expected_dtype)
                outputs_2.append(tmp.unsqueeze(dim=-2))
            output_2 = torch.cat(outputs_2, dim=-2)
            output_2 = torch.matmul(gating_distribution_2, output_2.to(gating_distribution_2.dtype)).squeeze(dim=-2)
            result += output_2

            return result



    class MoELinear4bitV5(bnb.nn.Linear4bit, LoraLayer):
        """ 4 * Con modules + 4 * Div modules """
        def __init__(
                self,
                adapter_name,
                in_features,
                out_features,
                r: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0.0,
                expert_nums=None,
                **kwargs,
        ) -> None:
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                compute_dtype=kwargs.get("compute_dtype", torch.float32),
                compress_statistics=kwargs.get("compress_statistics", True),
                quant_type=kwargs.get("quant_type", "nf4"),
            )
            LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)

            for suffix in ["A", "B", "C", "D", "E", "F", "G", "H"]:
                for net_suffix in ["Net1", "Net2"]:
                    attr_name = f"lora_{suffix}_{net_suffix}"
                    setattr(self, attr_name, nn.ModuleDict({}))

            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, expert_nums)

            for i in range(1, 9):
                router_attr_name = f"router_{i}"
                setattr(self, router_attr_name, Router(self.weight.shape[-1], expert_nums).to(self.weight.device))

            self.MoERouter = Router(self.weight.shape[-1], 8).to(self.weight.device)

            for p in chain(self.router_1.parameters(), self.router_2.parameters(), self.router_3.parameters(), self.router_4.parameters(),
                           self.router_5.parameters(), self.router_6.parameters(), self.router_7.parameters(), self.router_8.parameters()):
                p.data = p.data.to("cuda")
            for p in self.MoERouter.parameters():
                p.data = p.data.to("cuda")


        def merge(self):
            raise NotImplementedError

        def unmerge(self):
            raise NotImplementedError

        def get_delta_weight(self, adapter):
            raise NotImplementedError

        def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, expert_nums):
            if r <= 0:
                raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
            self.r[adapter_name] = r
            self.lora_alpha[adapter_name] = lora_alpha

            lora_dropout_layer = nn.Dropout(p=lora_dropout)
            self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

            # Actual trainable parameters
            if r > 0:
                self.lora_A_Net1[adapter_name] = nn.Linear(self.in_features, r, bias=False)
                for i in range(expert_nums):
                    self.lora_B_Net1[f"expert-{i}"] = nn.Linear(r, self.out_features, bias=False)

                for i in range(expert_nums):
                    self.lora_A_Net2[f"expert-{i}"] = nn.Linear(self.in_features, r, bias=False)
                self.lora_B_Net2[adapter_name] = nn.Linear(r, self.out_features, bias=False)

                self.lora_C_Net1[adapter_name] = nn.Linear(self.in_features, r, bias=False)
                for i in range(expert_nums):
                    self.lora_D_Net1[f"expert-{i}"] = nn.Linear(r, self.out_features, bias=False)

                for i in range(expert_nums):
                    self.lora_C_Net2[f"expert-{i}"] = nn.Linear(self.in_features, r, bias=False)
                self.lora_D_Net2[adapter_name] = nn.Linear(r, self.out_features, bias=False)

                self.lora_E_Net1[adapter_name] = nn.Linear(self.in_features, r, bias=False)
                for i in range(expert_nums):
                    self.lora_F_Net1[f"expert-{i}"] = nn.Linear(r, self.out_features, bias=False)

                for i in range(expert_nums):
                    self.lora_E_Net2[f"expert-{i}"] = nn.Linear(self.in_features, r, bias=False)
                self.lora_F_Net2[adapter_name] = nn.Linear(r, self.out_features, bias=False)

                self.lora_G_Net1[adapter_name] = nn.Linear(self.in_features, r, bias=False)
                for i in range(expert_nums):
                    self.lora_H_Net1[f"expert-{i}"] = nn.Linear(r, self.out_features, bias=False)

                for i in range(expert_nums):
                    self.lora_G_Net2[f"expert-{i}"] = nn.Linear(self.in_features, r, bias=False)
                self.lora_H_Net2[adapter_name] = nn.Linear(r, self.out_features, bias=False)

                self.scaling[adapter_name] = lora_alpha / r

            if init_lora_weights:
                self.reset_lora_parameters(adapter_name)

            weight = getattr(self, "weight", None)
            if weight is not None:
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)

        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            """
            [Shape of Intermediate variable in Llama-2-7B]
            x: torch.Size([1, 4096, 4096])
            mask: torch.Size([1, 4096, 8])
            gating_distribution_1: torch.Size([1, 4096, 1, 2])
            output_1: torch.Size([1, 4096, 4096])
            """
            def gating(input_x):
                return [router(input_x).unsqueeze(-2)
                        for router in (self.router_1, self.router_2, self.router_3, self.router_4,
                                       self.router_5, self.router_6, self.router_7, self.router_8)]

            # rank0_print(f"x: {x}", x.shape)
            mask = self.MoERouter(x)
            # rank0_print(f"mask: {mask}", mask.shape)
            mask_splits = torch.split(mask, split_size_or_sections=1, dim=-1)
            gating_1, gating_2, gating_3, gating_4, gating_5, gating_6, gating_7, gating_8 = gating(x)

            result = super().forward(x).clone()
            # rank0_print(f"result: {result}", x.shape)

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(self.lora_A_Net1["expert-0"].weight.dtype)

            # Process Network
            outputs_per_network = []

            # Network_1
            for net_idx, (lora_A_net, lora_B_net, net_gating) in enumerate([
                    (self.lora_A_Net1, self.lora_B_Net1, gating_1), (self.lora_C_Net1, self.lora_D_Net1, gating_3),
                    (self.lora_E_Net1, self.lora_F_Net1, gating_5), (self.lora_G_Net1, self.lora_H_Net1, gating_7)]):
                lora_A = lora_A_net["expert-0"]
                dropout = self.lora_dropout["expert-0"]
                scaling = self.scaling["expert-0"]
                outputs_1 = [lora_B(lora_A(dropout(x))) * scaling for active_adapter, lora_B in
                             list(lora_B_net.items())[:1]]
                outputs_1 = [output.unsqueeze(dim=-2) for output in outputs_1]
                for active_adapter, lora_B in list(lora_B_net.items())[1:]:
                    tmp = lora_B(lora_A(dropout(x))) * scaling
                    if requires_conversion:
                        tmp = tmp.to(expected_dtype)
                    outputs_1.append(tmp.unsqueeze(dim=-2))
                    # rank0_print("outputs_1:", outputs_1)
                net_output = torch.cat(outputs_1, dim=-2)
                net_output = torch.matmul(net_gating, net_output.to(net_gating.dtype)).squeeze(dim=-2)
                outputs_per_network.append(net_output)

            # Network_2
            for net_idx, (lora_A_net, lora_B_net, net_gating) in enumerate([
                    (self.lora_A_Net2, self.lora_B_Net2, gating_2), (self.lora_C_Net2, self.lora_D_Net2, gating_4),
                    (self.lora_E_Net2, self.lora_F_Net2, gating_6), (self.lora_G_Net2, self.lora_H_Net2, gating_8)]):
                lora_B = lora_B_net["expert-0"]
                dropout = self.lora_dropout["expert-0"]
                scaling = self.scaling["expert-0"]
                outputs_2 = [lora_B(lora_A(dropout(x))) * scaling for active_adapter, lora_A in
                             list(lora_A_net.items())[:1]]
                outputs_2 = [output.unsqueeze(dim=-2) for output in outputs_2]
                for active_adapter, lora_A in list(lora_A_net.items())[1:]:
                    tmp = lora_B(lora_A(dropout(x))) * scaling
                    if requires_conversion:
                        tmp = tmp.to(expected_dtype)
                    outputs_2.append(tmp.unsqueeze(dim=-2))
                    # rank0_print("outputs_2:", outputs_2)
                net_output = torch.cat(outputs_2, dim=-2)
                net_output = torch.matmul(net_gating, net_output.to(net_gating.dtype)).squeeze(dim=-2)
                outputs_per_network.append(net_output)

            selected_output = sum(output * mask_split for output, mask_split in zip(outputs_per_network, mask_splits))
            result += selected_output

            return result


