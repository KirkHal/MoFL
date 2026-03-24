# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils.other import transpose

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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = super().forward(x)
            elif self.merged:
                result = super().forward(x)
            else:
                result = super().forward(x)
                # As per Tim Dettmers, for 4bit, we need to defensively clone here.
                # The reason is that in some cases, an error can occur that backprop
                # does not work on a manipulated view. This issue may be solved with
                # newer PyTorch versions but this would need extensive testing to be
                # sure.
                result = result.clone()

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
                        x = x.to(lora_A.weight.dtype)

                    output = lora_B(lora_A(dropout(x)))
                    if requires_conversion:
                        output = output.to(expected_dtype)
                    output = output * scaling
                    result += output

            return result


    class MoELinear4bit(bnb.nn.Linear4bit, LoraLayer):
        # router在adapter前面
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            expert_nums = None,
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

            self.router = Router(self.weight.shape[0], 4, expert_nums).to(self.weight.device)  # 70B
            # self.router = Router(self.weight.shape[-1], 128, expert_nums).to(self.weight.device)  # 7B
            rank0_print("MoELinear4bit")



        def _gating(self, x,):
            weights = self.router(x)  # [batch_size, seq_len, experts_num]
            weights = F.softmax(weights, dim=-1).unsqueeze(dim=-2)  # [batch_size, seq_len, 1, experts_num]

            return weights
        

        def merge(self):
            raise NotImplementedError

        def unmerge(self):
            raise NotImplementedError

        def get_delta_weight(self, adapter):
            raise NotImplementedError

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # 1 router
            gating_distribution = self._gating(x,)

            result = super().forward(x)
            # As per Tim Dettmers, for 4bit, we need to defensively clone here.
            # The reason is that in some cases, an error can occur that backprop
            # does not work on a manipulated view. This issue may be solved with
            # newer PyTorch versions but this would need extensive testing to be
            # sure.
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
            
            return result


    class MoELinear4bitV2(bnb.nn.Linear4bit, LoraLayer):
        # 1. router在adapter后面
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            expert_nums = None,
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

            self.router = Router(self.weight.shape[0], 4, expert_nums).to(self.weight.device)  # 70B
            # self.router = Router(self.weight.shape[-1], 128, expert_nums).to(self.weight.device)  # 7B
            rank0_print("MoELinear4bitV2")



        def _gating(self, x,):
            weights = self.router(x)  # [batch_size, seq_len, experts_num]
            weights = F.softmax(weights, dim=-1).unsqueeze(dim=-2)  # [batch_size, seq_len, 1, experts_num]

            return weights
        
        def merge(self):
            raise NotImplementedError

        def unmerge(self):
            raise NotImplementedError

        def get_delta_weight(self, adapter):
            raise NotImplementedError

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x.shape == torch.Size([1, 4096, 8192])
            # gating_distribution = self._gating(x,)  # torch.Size([1, 4096, 1, 2])

            result = super().forward(x) # torch.Size([1, 4096, 8192])

            # As per Tim Dettmers, for 4bit, we need to defensively clone here.
            # The reason is that in some cases, an error can occur that backprop
            # does not work on a manipulated view. This issue may be solved with
            # newer PyTorch versions but this would need extensive testing to be
            # sure.
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
                output = torch.cat((output, tmp), dim=-2)   # torch.Size([1, 4096, 2, 8192])

            gating_distribution = self._gating(output.mean(dim=-2) * len(self.experts) + result)    # torch.Size([1, 4096, 1, 2])
            result += torch.matmul(gating_distribution.to(result.dtype), output.to(result.dtype)).squeeze(dim=-2)
            
            return result

    
    class MoELinear4bitV3(bnb.nn.Linear4bit, LoraLayer):
        # 1. router在adapter后面
        # 2. 只构建1个adapter，对其输出进行分割
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            expert_nums = None,
            **kwargs,
        ) -> None:
            self.expert_nums = expert_nums
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,  # 创建一个很大的adapter1
                bias=kwargs.get("bias", True),
                compute_dtype=kwargs.get("compute_dtype", torch.float32),
                compress_statistics=kwargs.get("compress_statistics", True),
                quant_type=kwargs.get("quant_type", "nf4"),
            )
            LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)

            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.router = Router(self.weight.shape[0], 4, expert_nums).to(self.weight.device)  # 70B
            for p in self.router.parameters():
                p.data = p.data.to("cuda")
            # self.router = Router(self.weight.shape[-1], 128, expert_nums).to(self.weight.device)  # 7B
            rank0_print("MoELinear4bitV3")



        def _gating(self, x,):
            weights = self.router(x)  # [batch_size, seq_len, experts_num]
            weights = F.softmax(weights, dim=-1).unsqueeze(dim=-2)  # [batch_size, seq_len, 1, experts_num]

            return weights
        

        def merge(self):
            raise NotImplementedError

        def unmerge(self):
            raise NotImplementedError

        def get_delta_weight(self, adapter):
            raise NotImplementedError

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            assert len(self.experts) == 1
            # x.shape == torch.Size([1, 4096, 8192])
            # gating_distribution = self._gating(x,)  # torch.Size([1, 4096, 1, 2])

            result = super().forward(x) # torch.Size([1, 4096, 8192])

            # As per Tim Dettmers, for 4bit, we need to defensively clone here.
            # The reason is that in some cases, an error can occur that backprop
            # does not work on a manipulated view. This issue may be solved with
            # newer PyTorch versions but this would need extensive testing to be
            # sure.
            result = result.clone()
            x = torch.cat([x] * self.expert_nums, dim=-1)

            active_adapter, = self.experts
                
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(lora_A.weight.dtype)

            output = lora_B(lora_A(dropout(x))) * scaling  # [batch_size, seq_len, emb_dim * expert_nums]
            output = output.unsqueeze(dim=-2)  # [batch_size, seq_len, 1, emb_dim * expert_nums]
            output = torch.cat(torch.split(output, int(output.shape[-1] / self.expert_nums), dim=-1), dim=-2)   # [batch_size, seq_len, 2, emb_dim]

            gating_distribution = self._gating(output.mean(dim=-2) * self.expert_nums + result)    # torch.Size([1, 4096, 1, 2])
            result += torch.matmul(gating_distribution.to(result.dtype), output.to(result.dtype)).squeeze(dim=-2)
            
            return result
        
        def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
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
                self.lora_A[adapter_name] = nn.Linear(self.in_features * self.expert_nums, r, bias=False)
                self.lora_B[adapter_name] = nn.Linear(r, self.out_features * self.expert_nums, bias=False)
                self.scaling[adapter_name] = lora_alpha / r
            if init_lora_weights:
                self.reset_lora_parameters(adapter_name)

            weight = getattr(self, "weight", None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)
            
            # 针对QLoRA
            if not hasattr(self, "experts"):
                self.experts = None

            if self.experts is None:
                setattr(self, "experts", [adapter_name])
                self.set_adapter(self.experts)
            else:
                self.experts.append(adapter_name)
                self.set_adapter(self.experts)



    class MoELinear4bitV4(bnb.nn.Linear4bit, LoraLayer):
        # 1. Router在Adapter后面
        # 2. 只构建1个adapter，对其输出进行分割
        # 3. 在保证参数量相等的情况下，把adapter数量进行扩充，例如6，60，600
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            expert_nums = None,
            **kwargs,
        ) -> None:
            self.expert_nums = expert_nums
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,  # 创建一个很大的adapter1
                bias=kwargs.get("bias", True),
                compute_dtype=kwargs.get("compute_dtype", torch.float32),
                compress_statistics=kwargs.get("compress_statistics", True),
                quant_type=kwargs.get("quant_type", "nf4"),
            )
            LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)

            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            # 在保证参数量不变的情况下，将expert数量扩充到6，60和600，因此这里的expert-num需要和update_layer对齐，只是拆的时候变成60份或600份
            # self.router = Router(self.weight.shape[0], 4, expert_nums).to(self.weight.device)  # 70B
            self.router = Router(int(self.weight.shape[0]), 4, expert_nums).to(self.weight.device).to("cuda")
            # self.router = Router(self.weight.shape[-1], 128, expert_nums).to(self.weight.device)  # 7B
            rank0_print("MoELinear4bitV4")



        def _gating(self, x,):
            weights = self.router(x)  # [batch_size, seq_len, experts_num]
            weights = F.softmax(weights, dim=-1).unsqueeze(dim=-2)  # [batch_size, seq_len, 1, experts_num]

            return weights
        

        def merge(self):
            raise NotImplementedError

        def unmerge(self):
            raise NotImplementedError

        def get_delta_weight(self, adapter):
            raise NotImplementedError

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            assert len(self.experts) == 1
            # x.shape == torch.Size([1, 4096, 8192])
            # gating_distribution = self._gating(x,)  # torch.Size([1, 4096, 1, 2])

            result = super().forward(x) # torch.Size([1, 4096, 8192])

            # As per Tim Dettmers, for 4bit, we need to defensively clone here.
            # The reason is that in some cases, an error can occur that backprop
            # does not work on a manipulated view. This issue may be solved with
            # newer PyTorch versions but this would need extensive testing to be
            # sure.
            result = result.clone()
            # 在保证参数量不变的情况下，将expert数量扩充到6，60和600，因此这里的expert-num需要和update_layer对齐，只是拆的时候变成60份或600份

            active_adapter, = self.experts
                
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(lora_A.weight.dtype)

            output = lora_B(lora_A(dropout(x))) * scaling  # [batch_size, seq_len, emb_dim]
            gating_distribution = self._gating(output.sum(dim=-2))   # torch.Size([1, 4096, 1, expert_nums])
            output = output.unsqueeze(dim=-2)  # [batch_size, seq_len, 1, emb_dim]
            # output = torch.cat(torch.split(output, int(output.shape[-1] / self.expert_nums), dim=-1), dim=-2)   # [batch_size, seq_len, expert_nums, exp_dim]  

            d1, d2, _, _ = output.shape
            original_shape = output.shape
            output = output.reshape((d1, d2, self.expert_nums, -1)) # [batch_size, seq_len, expert_nums, exp_dim]
            rank0_print("output: ", output.shape)
            rank0_print("gating_distribution: ", gating_distribution.shape)
            output = gating_distribution.transpose(-1, -2).expand(output.shape) * output    # [batch_size, seq_len, expert_nums, exp_dim]
            result = result + output.reshape(result.shape).to(result.dtype)
            
            return result
        
        def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
            # 在保证参数量不变的情况下，将expert数量扩充到6，60和600
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
                # self.lora_A[adapter_name] = nn.Linear(self.in_features * self.expert_nums, r, bias=False)
                # self.lora_B[adapter_name] = nn.Linear(r, self.out_features * self.expert_nums, bias=False)
                self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
                self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
                self.scaling[adapter_name] = lora_alpha / r
            if init_lora_weights:
                self.reset_lora_parameters(adapter_name)

            weight = getattr(self, "weight", None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)
            
            # 针对QLoRA
            if not hasattr(self, "experts"):
                self.experts = None

            if self.experts is None:
                setattr(self, "experts", [adapter_name])
                self.set_adapter(self.experts)
            else:
                self.experts.append(adapter_name)
                self.set_adapter(self.experts)



    class MoELinear4bitV5(bnb.nn.Linear4bit, LoraLayer):
        # 1. 2个Router在Adapter前面和后面
        # 2. 只构建1个adapter，对其输出进行分割
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            expert_nums = None,
            **kwargs,
        ) -> None:
            self.expert_nums = expert_nums
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,  # 创建一个很大的adapter1
                bias=kwargs.get("bias", True),
                compute_dtype=kwargs.get("compute_dtype", torch.float32),
                compress_statistics=kwargs.get("compress_statistics", True),
                quant_type=kwargs.get("quant_type", "nf4"),
            )
            LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)

            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            # 在保证参数量不变的情况下，将expert数量扩充到6，60和600，因此这里的expert-num需要和update_layer对齐，只是拆的时候变成60份或600份

            # self.router = Router(self.weight.shape[0], 4, expert_nums).to(self.weight.device)  # 70B
            self.router1 = Router(self.weight.shape[-1], 4, expert_nums).to(self.weight.device)
            self.router2 = Router(self.weight.shape[0], 4, expert_nums).to(self.weight.device)
            for p in self.router1.parameters():
                p.data = p.data.to("cuda")
            for p in self.router2.parameters():
                p.data = p.data.to("cuda")
            # self.router = Router(self.weight.shape[-1], 128, expert_nums).to(self.weight.device)  # 7B
            rank0_print("MoELinear4bitV5")



        def _gating_1(self, x,):
            weights = self.router1(x)  # [batch_size, seq_len, experts_num]
            weights = F.softmax(weights, dim=-1).unsqueeze(dim=-2)  # [batch_size, seq_len, 1, experts_num]

            return weights

        def _gating_2(self, x,):
            weights = self.router2(x)  # [batch_size, seq_len, experts_num]
            weights = F.softmax(weights, dim=-1).unsqueeze(dim=-2)  # [batch_size, seq_len, 1, experts_num]

            return weights
        

        def merge(self):
            raise NotImplementedError

        def unmerge(self):
            raise NotImplementedError

        def get_delta_weight(self, adapter):
            raise NotImplementedError

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            assert len(self.experts) == 1
            # x.shape == torch.Size([1, 4096, 8192])
            result = super().forward(x) # torch.Size([1, 4096, 8192])
            gating_distribution = self._gating_1(x,)  # torch.Size([1, 4096, 1, 2])

            # As per Tim Dettmers, for 4bit, we need to defensively clone here.
            # The reason is that in some cases, an error can occur that backprop
            # does not work on a manipulated view. This issue may be solved with
            # newer PyTorch versions but this would need extensive testing to be
            # sure.
            result = result.clone()

            x = torch.cat([x] * self.expert_nums, dim=-1)

            active_adapter, = self.experts
                
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(lora_A.weight.dtype)

            output = lora_B(lora_A(dropout(x))) * scaling  # [batch_size, seq_len, emb_dim * expert_nums]
            output = output.unsqueeze(dim=-2)  # [batch_size, seq_len, 1, emb_dim * expert_nums]
            output = torch.cat(torch.split(output, int(output.shape[-1] / self.expert_nums), dim=-1), dim=-2)   # [batch_size, seq_len, 2, emb_dim]
            output = (gating_distribution.transpose(-1, -2).expand(tuple(output.shape))) * output

            gating_distribution_2 = self._gating_2(output.sum(dim=-2))    # torch.Size([1, 4096, 1, 2])
            result += torch.matmul(gating_distribution_2.to(result.dtype), output.to(result.dtype)).squeeze(dim=-2)
            
            return result
        
        def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
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
                self.lora_A[adapter_name] = nn.Linear(self.in_features * self.expert_nums, r, bias=False)
                self.lora_B[adapter_name] = nn.Linear(r, self.out_features * self.expert_nums, bias=False)
                self.scaling[adapter_name] = lora_alpha / r
            if init_lora_weights:
                self.reset_lora_parameters(adapter_name)

            weight = getattr(self, "weight", None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)
            
            # 针对QLoRA
            if not hasattr(self, "experts"):
                self.experts = None

            if self.experts is None:
                setattr(self, "experts", [adapter_name])
                self.set_adapter(self.experts)
            else:
                self.experts.append(adapter_name)
                self.set_adapter(self.experts)


    class MoELinear4bitV6(bnb.nn.Linear4bit, LoraLayer):
        # 1. router在adapter前面
        # 2. 只构建1个adapter，对其输出进行分割
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            expert_nums = None,
            **kwargs,
        ) -> None:
            self.expert_nums = expert_nums
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,  # 创建一个很大的adapter1
                bias=kwargs.get("bias", True),
                compute_dtype=kwargs.get("compute_dtype", torch.float32),
                compress_statistics=kwargs.get("compress_statistics", True),
                quant_type=kwargs.get("quant_type", "nf4"),
            )
            LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)

            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.router = Router(self.weight.shape[-1], 4, expert_nums).to(self.weight.device)  # 70B
            for p in self.router.parameters():
                p.data = p.data.to("cuda")
            # self.router = Router(self.weight.shape[-1], 128, expert_nums).to(self.weight.device)  # 7B
            rank0_print("MoELinear4bitV6")



        def _gating(self, x,):
            weights = self.router(x)  # [batch_size, seq_len, experts_num]
            weights = F.softmax(weights, dim=-1).unsqueeze(dim=-2)  # [batch_size, seq_len, 1, experts_num]

            return weights
        

        def merge(self):
            raise NotImplementedError

        def unmerge(self):
            raise NotImplementedError

        def get_delta_weight(self, adapter):
            raise NotImplementedError

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            assert len(self.experts) == 1
            # x.shape == torch.Size([1, 4096, 8192])
            gating_distribution = self._gating(x,)  # torch.Size([1, 4096, 1, 2])

            result = super().forward(x) # torch.Size([1, 4096, 8192])

            # As per Tim Dettmers, for 4bit, we need to defensively clone here.
            # The reason is that in some cases, an error can occur that backprop
            # does not work on a manipulated view. This issue may be solved with
            # newer PyTorch versions but this would need extensive testing to be
            # sure.
            result = result.clone()
            x = torch.cat([x] * self.expert_nums, dim=-1)

            active_adapter, = self.experts
                
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(lora_A.weight.dtype)

            output = lora_B(lora_A(dropout(x))) * scaling  # [batch_size, seq_len, emb_dim * expert_nums]
            output = output.unsqueeze(dim=-2)  # [batch_size, seq_len, 1, emb_dim * expert_nums]
            output = torch.cat(torch.split(output, int(output.shape[-1] / self.expert_nums), dim=-1), dim=-2)   # [batch_size, seq_len, 2, emb_dim]

            # gating_distribution = self._gating(output.mean(dim=-2) * self.expert_nums + result)    # torch.Size([1, 4096, 1, 2])
            result += torch.matmul(gating_distribution.to(result.dtype), output.to(result.dtype)).squeeze(dim=-2)
            
            return result
        
        def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
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
                self.lora_A[adapter_name] = nn.Linear(self.in_features * self.expert_nums, r, bias=False)
                self.lora_B[adapter_name] = nn.Linear(r, self.out_features * self.expert_nums, bias=False)
                self.scaling[adapter_name] = lora_alpha / r
            if init_lora_weights:
                self.reset_lora_parameters(adapter_name)

            weight = getattr(self, "weight", None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)
            
            # 针对QLoRA
            if not hasattr(self, "experts"):
                self.experts = None

            if self.experts is None:
                setattr(self, "experts", [adapter_name])
                self.set_adapter(self.experts)
            else:
                self.experts.append(adapter_name)
                self.set_adapter(self.experts)