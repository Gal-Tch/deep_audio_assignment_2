import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os

def collapse_sequence(sequence):
    """
    Implements the CTC collapse function B.
    1. Removes consecutive repeating characters.
    2. Removes the blank symbol '^'.
    """
    if not sequence:
        return ""
    
    # Remove consecutive repeats
    no_repeats = []
    if len(sequence) > 0:
        no_repeats.append(sequence[0])
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i-1]:
                no_repeats.append(sequence[i])
            
    # Remove blanks
    collapsed = [char for char in no_repeats if char != '^']
    return "".join(collapsed)

def logsumexp(probs):
    """Helper to compute log(sum(exp(probs))) safely."""
    max_p = np.max(probs)
    if max_p == -np.inf:
        return -np.inf
    return max_p + np.log(np.sum(np.exp(probs - max_p)))

def ctc_forward_pass(pred, sequence, alphabet_map, mode='sum', use_log=False):
    """
    Implements the CTC forward pass.
    pred: T x V matrix of probabilities (T frames, V vocabulary size)
    sequence: The target string (e.g., 'aba')
    alphabet_map: Dictionary mapping index to character (including blank)
    mode: 'sum' for Forward Algorithm, 'max' for Force Alignment (Viterbi-like)
    use_log: If True, perform calculations in log-space to avoid underflow.
    """
    blank = '^'
    s_prime = [blank]
    for char in sequence:
        s_prime.append(char)
        s_prime.append(blank)
    
    T = pred.shape[0]
    S = len(s_prime)
    
    # Initialize alpha table
    if use_log:
        alpha = np.full((T, S), -np.inf)
        log_pred = np.log(pred + 1e-30) # Avoid log(0)
    else:
        alpha = np.zeros((T, S))
    
    # Map alphabet characters to indices
    char_to_idx = {v: k for k, v in alphabet_map.items()}
    
    # Initial probabilities at t=0
    if use_log:
        if s_prime[0] in char_to_idx:
            alpha[0, 0] = log_pred[0, char_to_idx[s_prime[0]]]
        if s_prime[1] in char_to_idx:
            alpha[0, 1] = log_pred[0, char_to_idx[s_prime[1]]]
    else:
        if s_prime[0] in char_to_idx:
            alpha[0, 0] = pred[0, char_to_idx[s_prime[0]]]
        if s_prime[1] in char_to_idx:
            alpha[0, 1] = pred[0, char_to_idx[s_prime[1]]]
    
    # Dynamic Programming
    for t in range(1, T):
        for s in range(S):
            curr_char = s_prime[s]
            if curr_char not in char_to_idx:
                continue
                
            if use_log:
                p_curr = log_pred[t, char_to_idx[curr_char]]
                options = [alpha[t-1, s]]
                if s > 0:
                    options.append(alpha[t-1, s-1])
                if s >= 2 and curr_char != blank and curr_char != s_prime[s-2]:
                    options.append(alpha[t-1, s-2])
                
                if mode == 'sum':
                    alpha[t, s] = logsumexp(options) + p_curr
                else:
                    alpha[t, s] = np.max(options) + p_curr
            else:
                p_curr = pred[t, char_to_idx[curr_char]]
                options = [alpha[t-1, s]]
                if s > 0:
                    options.append(alpha[t-1, s-1])
                if s >= 2 and curr_char != blank and curr_char != s_prime[s-2]:
                    options.append(alpha[t-1, s-2])
                
                if mode == 'sum':
                    alpha[t, s] = np.sum(options) * p_curr
                else:
                    alpha[t, s] = np.max(options) * p_curr
                
    return alpha

def question_5():
    print("\n--- Question 5: Forward Pass ---")
    pred = np.zeros(shape=(5, 3), dtype=np.float32)
    pred[0][0], pred[0][1] = 0.8, 0.2
    pred[1][0], pred[1][1] = 0.2, 0.8
    pred[2][0], pred[2][1] = 0.3, 0.7
    pred[3][0], pred[3][1], pred[3][2] = 0.09, 0.8, 0.11
    pred[4][2] = 1.00
    
    alphabet_map = {0: 'a', 1: 'b', 2: '^'}
    
    alpha = ctc_forward_pass(pred, 'aba', alphabet_map, mode='sum', use_log=False)
    total_prob = alpha[-1, -1] + alpha[-1, -2]
    print(f"Total probability of sequence 'aba': {total_prob:.5f}")
    
    plt.figure(figsize=(8, 5))
    plt.imshow(pred.T, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Probability')
    plt.yticks(range(3), [alphabet_map[i] for i in range(3)])
    plt.xlabel('Time Frame (t)')
    plt.ylabel('Character')
    plt.title('Question 5: Prediction Matrix')
    plt.savefig('results/q5_pred_matrix.png')
    plt.close()
    
    return pred, alphabet_map

def question_6(pred, alphabet_map):
    print("\n--- Question 6: Force Alignment (aba) ---")
    alpha_max = ctc_forward_pass(pred, 'aba', alphabet_map, mode='max', use_log=False)
    
    T, S = alpha_max.shape
    blank = '^'
    sequence = 'aba'
    s_prime = ['^']
    for char in sequence:
        s_prime.append(char)
        s_prime.append('^')
        
    path_indices = []
    curr_s = np.argmax(alpha_max[T-1, :])
    path_indices.append(curr_s)
    
    for t in range(T-2, -1, -1):
        prev_options = [(alpha_max[t, curr_s], curr_s)]
        if curr_s > 0:
            prev_options.append((alpha_max[t, curr_s-1], curr_s-1))
        if curr_s >= 2 and s_prime[curr_s] != blank and s_prime[curr_s] != s_prime[curr_s-2]:
            prev_options.append((alpha_max[t, curr_s-2], curr_s-2))
            
        curr_s = max(prev_options, key=lambda x: x[0])[1]
        path_indices.append(curr_s)
        
    path_indices.reverse()
    best_path = [s_prime[idx] for idx in path_indices]
    
    print(f"Most probable path for 'aba': {best_path}")
    print(f"Collapsed result: {collapse_sequence(best_path)}")
    print(f"Path probability: {np.max(alpha_max[-1, :]):.5f}")
    
    char_to_idx = {v: k for k, v in alphabet_map.items()}
    path_y_coords = [char_to_idx[char] for char in best_path]
    
    plt.figure(figsize=(8, 5))
    plt.imshow(pred.T, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.plot(range(T), path_y_coords, color='red', marker='o', linewidth=2, label='Best Path')
    plt.colorbar(label='Probability')
    plt.yticks(range(3), [alphabet_map[i] for i in range(3)])
    plt.xlabel('Time Frame (t)')
    plt.ylabel('Character')
    plt.title('Question 6: Force Alignment with Best Path')
    plt.legend()
    plt.savefig('results/q6_force_alignment.png')
    plt.close()

def question_7():
    print("\n--- Question 7: Real Data Force Alignment ---")
    if not os.path.exists('force_align.pkl'):
        print("Error: force_align.pkl not found!")
        return

    with open('force_align.pkl', 'rb') as f:
        data = pkl.load(f)
    
    probs = data['acoustic_model_out_probs']
    alphabet_map = data['label_mapping']
    text_to_align = data['text_to_align']
    gt_text = data['gt_text']
    
    print(f"Ground Truth: '{gt_text}'")
    print(f"Text to Align: '{text_to_align}'")
    
    # Use log-space for real data to avoid underflow (T is larger)
    alpha_max = ctc_forward_pass(probs, text_to_align, alphabet_map, mode='max', use_log=True)
    
    T, S = alpha_max.shape
    blank = '^'
    s_prime = [blank]
    for char in text_to_align:
        s_prime.append(char)
        s_prime.append(blank)
    
    # Backtrack
    path_indices = []
    curr_s = np.argmax(alpha_max[T-1, :])
    path_indices.append(curr_s)
    
    for t in range(T-2, -1, -1):
        prev_options = [(alpha_max[t, curr_s], curr_s)]
        if curr_s > 0:
            prev_options.append((alpha_max[t, curr_s-1], curr_s-1))
        if curr_s >= 2 and s_prime[curr_s] != blank and s_prime[curr_s] != s_prime[curr_s-2]:
            prev_options.append((alpha_max[t, curr_s-2], curr_s-2))
            
        curr_s = max(prev_options, key=lambda x: x[0])[1]
        path_indices.append(curr_s)
    
    path_indices.reverse()
    best_path = [s_prime[idx] for idx in path_indices]
    collapsed = collapse_sequence(best_path)
    
    print(f"Force Aligned Result: '{collapsed}'")
    
    # Plotting (probs.T is 29 x T)
    plt.figure(figsize=(15, 8))
    plt.imshow(probs.T, aspect='auto', interpolation='nearest', cmap='magma')
    
    # Only show non-zero path points on plot if possible
    char_to_idx = {v: k for k, v in alphabet_map.items()}
    path_y_coords = [char_to_idx[char] for char in best_path]
    plt.plot(range(T), path_y_coords, color='cyan', alpha=0.5, label='Alignment Path')
    
    plt.colorbar(label='Probability')
    # Filter y-ticks to show only characters present in the text to align for clarity
    relevant_chars = set(text_to_align) | {blank}
    relevant_indices = [i for i, c in alphabet_map.items() if c in relevant_chars]
    plt.yticks(relevant_indices, [alphabet_map[i] for i in relevant_indices])
    
    plt.xlabel('Time Frame (t)')
    plt.ylabel('Character')
    plt.title('Question 7: Force Alignment on Real Data')
    plt.savefig('results/q7_force_alignment_real.png')
    plt.close()

if __name__ == "__main__":
    if not os.path.exists('results'):
        os.makedirs('results')
        
    # Question 4
    test_seq = ['a', 'a', '^', '^', 'b', 'b', '^', 'a']
    print(f"Question 4: Collapse {test_seq} -> {collapse_sequence(test_seq)}")
    
    # Questions 5 & 6
    p, a = question_5()
    question_6(p, a)
    
    # Question 7
    question_7()
