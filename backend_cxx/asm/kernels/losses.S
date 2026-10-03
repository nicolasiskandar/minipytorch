  .text
  .globl nn_mean_squared_error_f64
  .type nn_mean_squared_error_f64, @function

# void nn_mean_squared_error_f64(const double* predicted, const double* target,
#   double* gradient, size_t count, double* loss)
nn_mean_squared_error_f64:
  xorpd %xmm0, %xmm0
  xorq  %rax, %rax
.Lmse_loop:
  cmpq  %rcx, %rax
  jae   .Lmse_done
  movsd (%rdi, %rax, 8), %xmm1
  subsd (%rsi, %rax, 8), %xmm1
  movsd %xmm1, (%rdx, %rax, 8)
  mulsd %xmm1, %xmm1
  mulsd .Lhalf(%rip), %xmm1
  addsd %xmm1, %xmm0
  incq  %rax
  jmp   .Lmse_loop
.Lmse_done:
  movsd %xmm0, (%r8)
  ret

  .section .rodata
.Lhalf:
  .double 0.5

  .section .note.GNU-stack,"",@progbits
