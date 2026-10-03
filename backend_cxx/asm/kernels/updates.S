  .text
  .globl nn_apply_gradients_f64
  .type nn_apply_gradients_f64, @function

# void nn_apply_gradients_f64(double* values, const double* gradients,
#                             size_t count, double learningRate)
# Arguments: rdi=values, rsi=gradients, rdx=count, xmm0=learningRate.
nn_apply_gradients_f64:
  xorq  %rax, %rax
.Lapply_gradients_loop:
  cmpq  %rdx, %rax
  jae   .Lapply_gradients_done
  movsd (%rsi, %rax, 8), %xmm1
  mulsd %xmm0, %xmm1
  movsd (%rdi, %rax, 8), %xmm2
  subsd %xmm1, %xmm2
  movsd %xmm2, (%rdi, %rax, 8)
  incq  %rax
  jmp   .Lapply_gradients_loop
.Lapply_gradients_done:
  ret

  .globl nn_accumulate_f64
  .type nn_accumulate_f64, @function

# void nn_accumulate_f64(double* destination, const double* source,
#                        size_t count)
nn_accumulate_f64:
  xorq  %rax, %rax
.Laccumulate_loop:
  cmpq  %rdx, %rax
  jae   .Laccumulate_done
  movsd (%rsi, %rax, 8), %xmm0
  addsd (%rdi, %rax, 8), %xmm0
  movsd %xmm0, (%rdi, %rax, 8)
  incq  %rax
  jmp   .Laccumulate_loop
.Laccumulate_done:
  ret

  .section .note.GNU-stack,"",@progbits
