  .text

  .globl nn_relu_f64
  .type nn_relu_f64, @function
nn_relu_f64:
  xorpd  %xmm1, %xmm1
  ucomisd %xmm1, %xmm0
  ja     .Lrelu_done
  xorpd  %xmm0, %xmm0
.Lrelu_done:
  ret

  .globl nn_relu_derivative_from_output_f64
  .type nn_relu_derivative_from_output_f64, @function
nn_relu_derivative_from_output_f64:
  xorpd  %xmm1, %xmm1
  ucomisd %xmm1, %xmm0
  ja     .Lrelu_derivative_positive
  xorpd  %xmm0, %xmm0
  ret
.Lrelu_derivative_positive:
  movsd .Lone(%rip), %xmm0
  ret

# Range-reduce value = k * ln(2) + r, then evaluate exp(r) with a ninth-degree
# Taylor polynomial. The intended input domain is [-16, 16].
  .globl nn_exp_approx_f64
  .type nn_exp_approx_f64, @function
nn_exp_approx_f64:
  movapd %xmm0, %xmm1
  mulsd  .Linv_ln2(%rip), %xmm1
  cvtsd2si %xmm1, %rax
  cvtsi2sd %rax, %xmm2
  mulsd  .Lln2(%rip), %xmm2
  subsd  %xmm2, %xmm0
  movsd .Lexp_c9(%rip), %xmm1
  mulsd %xmm0, %xmm1
  addsd .Lexp_c8(%rip), %xmm1
  mulsd %xmm0, %xmm1
  addsd .Lexp_c7(%rip), %xmm1
  mulsd %xmm0, %xmm1
  addsd .Lexp_c6(%rip), %xmm1
  mulsd %xmm0, %xmm1
  addsd .Lexp_c5(%rip), %xmm1
  mulsd %xmm0, %xmm1
  addsd .Lexp_c4(%rip), %xmm1
  mulsd %xmm0, %xmm1
  addsd .Lexp_c3(%rip), %xmm1
  mulsd %xmm0, %xmm1
  addsd .Lexp_c2(%rip), %xmm1
  mulsd %xmm0, %xmm1
  addsd .Lone(%rip), %xmm1
  mulsd %xmm0, %xmm1
  addsd .Lone(%rip), %xmm1
  addq $1023, %rax
  shlq $52, %rax
  movq %rax, %xmm2
  mulsd %xmm2, %xmm1
  movapd %xmm1, %xmm0
  ret

  .globl nn_sigmoid_f64
  .type nn_sigmoid_f64, @function
nn_sigmoid_f64:
  ucomisd %xmm0, %xmm0
  jp      .Lsigmoid_compute
  ucomisd .Lsigmoid_upper(%rip), %xmm0
  jae     .Lsigmoid_one
  ucomisd .Lsigmoid_lower(%rip), %xmm0
  jbe     .Lsigmoid_zero
.Lsigmoid_compute:
  xorpd  %xmm1, %xmm1
  subsd  %xmm0, %xmm1
  movapd %xmm1, %xmm0
  subq   $8, %rsp
  call   nn_exp_approx_f64
  addq   $8, %rsp
  addsd  .Lone(%rip), %xmm0
  movsd  .Lone(%rip), %xmm1
  divsd  %xmm0, %xmm1
  movapd %xmm1, %xmm0
  ret
.Lsigmoid_one:
  movsd .Lone(%rip), %xmm0
  ret
.Lsigmoid_zero:
  xorpd %xmm0, %xmm0
  ret

  .globl nn_sigmoid_derivative_from_output_f64
  .type nn_sigmoid_derivative_from_output_f64, @function
nn_sigmoid_derivative_from_output_f64:
  movsd .Lone(%rip), %xmm1
  subsd %xmm0, %xmm1
  mulsd %xmm1, %xmm0
  ret

  .globl nn_tanh_f64
  .type nn_tanh_f64, @function
nn_tanh_f64:
  ucomisd %xmm0, %xmm0
  jp      .Ltanh_nan
  ucomisd .Ltanh_upper(%rip), %xmm0
  jae     .Ltanh_one
  ucomisd .Ltanh_lower(%rip), %xmm0
  jbe     .Ltanh_negative_one
  addsd   %xmm0, %xmm0
  subq    $8, %rsp
  call    nn_exp_approx_f64
  addq    $8, %rsp
  movapd  %xmm0, %xmm1
  subsd   .Lone(%rip), %xmm1
  addsd   .Lone(%rip), %xmm0
  divsd   %xmm0, %xmm1
  movapd  %xmm1, %xmm0
  ret
.Ltanh_one:
  movsd .Lone(%rip), %xmm0
  ret
.Ltanh_negative_one:
  movsd .Lnegative_one(%rip), %xmm0
  ret
.Ltanh_nan:
  subsd %xmm0, %xmm0
  ret

  .globl nn_tanh_derivative_from_output_f64
  .type nn_tanh_derivative_from_output_f64, @function
nn_tanh_derivative_from_output_f64:
  mulsd %xmm0, %xmm0
  movsd .Lone(%rip), %xmm1
  subsd %xmm0, %xmm1
  movapd %xmm1, %xmm0
  ret

  .section .rodata
.p2align 3
.Lone: .double 1.0
.Lnegative_one: .double -1.0
.Lln2: .double 0.69314718055994530942
.Linv_ln2: .double 1.4426950408889634074
.Lsigmoid_lower: .double -14.0
.Lsigmoid_upper: .double 14.0
.Ltanh_lower: .double -8.0
.Ltanh_upper: .double 8.0
.Lexp_c2: .double 0.5
.Lexp_c3: .double 0.16666666666666666667
.Lexp_c4: .double 0.041666666666666666667
.Lexp_c5: .double 0.0083333333333333333333
.Lexp_c6: .double 0.0013888888888888888889
.Lexp_c7: .double 0.00019841269841269841270
.Lexp_c8: .double 0.000024801587301587301587
.Lexp_c9: .double 0.0000027557319223985890653

  .section .note.GNU-stack,"",@progbits
