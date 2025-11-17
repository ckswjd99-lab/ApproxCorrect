import torch
from typing import Tuple, Dict

def softmax_customed(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """S(t)에서 C와 D(t)를 계산 (캐시 생성)"""
    # C_i = max_k(S(t)_ik)
    max_input = torch.max(input, dim=-1, keepdim=True).values
    exp_input = torch.exp(input - max_input)
    # D(t)_i = sum_j(exp(S(t)_ij - C_i))
    sum_exp_input = torch.sum(exp_input, dim=-1, keepdim=True)
    softmax_output = exp_input / (sum_exp_input + 1e-6)
    # P(t), C, D(t) 반환
    return softmax_output, max_input, sum_exp_input

def cache_initial_softmax(attn_score_old: torch.Tensor, 
                            q_indices: torch.Tensor, 
                            k_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    1단계: 초기 스코어(S(t))에서 Softmax를 계산하고,
    '델타 업데이트'에 필요한 값들을 캐싱합니다.
    """
    # 1. S(t)로 C, D(t)를 계산
    # _, C, D(t)
    _, sm_max, sm_expsum = softmax_customed(attn_score_old)
    
    # 2. (I, J) 인덱스 생성
    q_idx_mesh, k_idx_mesh = torch.meshgrid(q_indices, k_indices, indexing='ij')

    # 3. S_sub(t) 추출 (i in I, j in J)
    attn_score_sampled_old = attn_score_old[:, q_idx_mesh, k_idx_mesh]
    
    # 4. 캐시 딕셔너리 생성
    cache = {
        "sm_max": sm_max,                 # C (전체 행에 대한 오프셋)
        "sm_expsum": sm_expsum,           # D(t) (전체 행에 대한 분모)
        "attn_score_sampled_old": attn_score_sampled_old, # S_sub(t)
        "q_indices": q_indices          # I (sm_max, sm_expsum 인덱싱용)
    }
    return cache

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

# 1. 기본 설정
B, DQ, DK = 12, 4096, 4096
SUB_DQ, SUB_DK = 1024, 1024
# torch.manual_seed(0)

# S(t)
attn_score_old = torch.randn(B, DQ, DK)

# I, J
rand_q_idx = torch.randperm(DQ)[:SUB_DQ]
rand_k_idx = torch.randperm(DK)[:SUB_DK]

# 2. 1단계 실행: S(t) -> 캐시 생성
cache = cache_initial_softmax(attn_score_old, rand_q_idx, rand_k_idx)

# 3. S_sub(t+1) 생성
attn_score_sampled_new = torch.randn(B, SUB_DQ, SUB_DK)

# 4. 2단계 실행: (캐시, S_sub(t+1)) -> P_sub(t+1)
attn_prob_sampled_new_delta = correct_partial_softmax(attn_score_sampled_new, cache)


# 5. "Ground Truth" 계산 (비교용)
attn_score_new = attn_score_old.clone()
q_idx_mesh, k_idx_mesh = torch.meshgrid(rand_q_idx, rand_k_idx, indexing='ij')
attn_score_new[:, q_idx_mesh, k_idx_mesh] = attn_score_sampled_new
attn_prob_new_gt_full = torch.softmax(attn_score_new, dim=-1)
attn_prob_new_gt = attn_prob_new_gt_full[:, q_idx_mesh, k_idx_mesh]


# 6. 비교
print(f"--- Ground Truth (torch.softmax) [0,0,:5] ---")
print(attn_prob_new_gt[0, 0, :5])

print(f"\n--- Delta Update (correct_partial_softmax) [0,0,:5] ---")
print(attn_prob_sampled_new_delta[0, 0, :5])

max_diff = torch.abs(attn_prob_new_gt - attn_prob_sampled_new_delta).max().item()
print(f"\n--- Max Difference ---")
print(max_diff)

if max_diff < 1e-6:
    print("\n✅ 성공: 'correct_partial_softmax' 함수가 'Ground Truth'와 일치합니다.")
else:
    print(f"\n❌ 실패: 로직이 일치하지 않습니다. (Diff: {max_diff})")