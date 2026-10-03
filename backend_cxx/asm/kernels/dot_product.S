  .text
  .globl nn_dot_product_f64
  .type nn_dot_product_f64, @function

# double nn_dot_product_f64(const double* left, const double* right,
#                           std::size_t length)
# Arguments: rdi = left, rsi = right, rdx = length. Result: xmm0.
nn_dot_product_f64:
  xorpd %xmm0, %xmm0
  xorq  %rax, %rax
  testq %rdx, %rdx
  je    .Ldot_done

.Ldot_loop:
  movsd (%rdi, %rax, 8), %xmm1
  mulsd (%rsi, %rax, 8), %xmm1
  addsd %xmm1, %xmm0
  incq  %rax
  cmpq  %rdx, %rax
  jne   .Ldot_loop

.Ldot_done:
  ret

  .section .note.GNU-stack,"",@progbits
