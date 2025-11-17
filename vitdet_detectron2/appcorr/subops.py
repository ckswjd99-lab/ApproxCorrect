import torch

from typing import Tuple, Dict


def softmax_customed(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_input = torch.max(input, dim=-1, keepdim=True).values
    exp_input = torch.exp(input - max_input)
    sum_exp_input = torch.sum(exp_input, dim=-1, keepdim=True)
    softmax_output = exp_input / sum_exp_input

    return softmax_output, max_input, sum_exp_input

def correct_partial_softmax(attn_score_sampled_new: torch.Tensor, 
                              cache: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    2단계: 캐시된 값과 S_sub(t+1)을 이용해 '델타 업데이트'를 수행하고,
    P_sub(t+1)을 반환합니다.
    """
    # 1. 캐시에서 필요한 값 로드
    sm_max = cache["sm_max"]
    sm_expsum = cache["sm_expsum"]
    attn_score_sampled_old = cache["attn_score_sampled_old"]
    q_indices = cache["q_indices"]
    
    # 2. 수식에 필요한 값들 추출
    # C_i (i in I)
    sm_max_sampled = sm_max[:, q_indices, :]
    # D(t)_i (i in I)
    sm_expsum_sampled_old = sm_expsum[:, q_indices, :]
    
    # === 수식 구현 시작 ===

    # 3. 분모 변화량 (Delta D_i) 계산
    # exp(S(t+1)_ik - C_i) for k in J
    exp_new = (attn_score_sampled_new - sm_max_sampled).exp()
    # exp(S(t)_ik - C_i) for k in J
    exp_old = (attn_score_sampled_old - sm_max_sampled).exp()

    # Delta D_i = sum_{k in J} ( exp_new - exp_old )
    d_sm_expsum_elements = exp_new - exp_old
    d_sm_expsum_sum = d_sm_expsum_elements.sum(dim=-1, keepdim=True)

    # 4. 새 분모 (D(t+1)_i) 계산
    # D(t+1)_i = D(t)_i + Delta D_i
    sm_expsum_sampled_new = sm_expsum_sampled_old + d_sm_expsum_sum

    # 5. 새 분자 계산
    # exp(S(t+1)_ij - C_i) for i in I, j in J
    numerator_new = exp_new

    # 6. 최종 확률 P_sub(t+1) 계산
    # P(t+1)_ij = 새 분자 / 새 분모
    attn_prob_sampled_new = numerator_new / (sm_expsum_sampled_new + 1e-6)

    # === 수식 구현 끝 ===
    
    return attn_prob_sampled_new