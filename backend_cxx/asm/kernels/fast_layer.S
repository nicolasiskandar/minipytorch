  .text
  .globl nn_fast_layer_forward_f64
  .type nn_fast_layer_forward_f64, @function

# The seventh argument, activationKind, is passed on the stack under the
# System V AMD64 ABI. Built-in activation kinds are 0=sigmoid, 1=tanh, 2=ReLU.
nn_fast_layer_forward_f64:
  pushq %rbx
  pushq %r12
  pushq %r13
  pushq %r14
  pushq %r15
  subq  $32, %rsp
  movq  %rdi, %r12
  movq  %rsi, %r13
  movq  %rdx, %r14
  movq  %rcx, %r15
  movq  %r8, %rbx
  movq  %r9, 0(%rsp)
  movl  80(%rsp), %eax
  movl  %eax, 8(%rsp)
  movq  $0, 16(%rsp)

.Lfast_forward_loop:
  movq  16(%rsp), %rax
  cmpq  0(%rsp), %rax
  jae   .Lfast_forward_done
  imulq %rbx, %rax
  leaq  (%r13, %rax, 8), %rdi
  movq  %r12, %rsi
  movq  %rbx, %rdx
  call  nn_dot_product_f64
  movq  16(%rsp), %rax
  addsd (%r14, %rax, 8), %xmm0
  cmpl  $0, 8(%rsp)
  je    .Lfast_forward_sigmoid
  cmpl  $1, 8(%rsp)
  je    .Lfast_forward_tanh
  call  nn_relu_f64
  jmp   .Lfast_forward_store
.Lfast_forward_sigmoid:
  call  nn_sigmoid_f64
  jmp   .Lfast_forward_store
.Lfast_forward_tanh:
  call  nn_tanh_f64
.Lfast_forward_store:
  movq  16(%rsp), %rax
  movsd %xmm0, (%r15, %rax, 8)
  incq  %rax
  movq  %rax, 16(%rsp)
  jmp   .Lfast_forward_loop

.Lfast_forward_done:
  addq  $32, %rsp
  popq  %r15
  popq  %r14
  popq  %r13
  popq  %r12
  popq  %rbx
  ret

  .globl nn_fast_layer_backward_f64
  .type nn_fast_layer_backward_f64, @function

# The first six arguments are in registers. The remaining arguments are:
# 7=dLoss_dInput, 8=numInputs, 9=numOutputs, 10=activationKind.
nn_fast_layer_backward_f64:
  pushq %rbx
  pushq %r12
  pushq %r13
  pushq %r14
  pushq %r15
  subq  $64, %rsp
  movq  %rdi, %r12
  movq  %rsi, %r13
  movq  %rdx, %r14
  movq  %rcx, %r15
  movq  %r8, 0(%rsp)
  movq  %r9, 8(%rsp)
  movq  112(%rsp), %rax
  movq  %rax, 16(%rsp)
  movq  120(%rsp), %rbx
  movq  128(%rsp), %rax
  movq  %rax, 24(%rsp)
  movl  136(%rsp), %eax
  movl  %eax, 32(%rsp)

  movq  $0, 48(%rsp)
.Lfast_backward_zero_input:
  movq  48(%rsp), %rax
  cmpq  %rbx, %rax
  jae   .Lfast_backward_outputs
  movq  16(%rsp), %r10
  movq  $0, (%r10, %rax, 8)
  incq  %rax
  movq  %rax, 48(%rsp)
  jmp   .Lfast_backward_zero_input

.Lfast_backward_outputs:
  movq  $0, 40(%rsp)
.Lfast_backward_output_loop:
  movq  40(%rsp), %rax
  cmpq  24(%rsp), %rax
  jae   .Lfast_backward_done
  movsd (%r14, %rax, 8), %xmm0
  cmpl  $0, 32(%rsp)
  je    .Lfast_backward_sigmoid
  cmpl  $1, 32(%rsp)
  je    .Lfast_backward_tanh
  call  nn_relu_derivative_from_output_f64
  jmp   .Lfast_backward_delta
.Lfast_backward_sigmoid:
  call  nn_sigmoid_derivative_from_output_f64
  jmp   .Lfast_backward_delta
.Lfast_backward_tanh:
  call  nn_tanh_derivative_from_output_f64
.Lfast_backward_delta:
  movq  40(%rsp), %rax
  mulsd (%r15, %rax, 8), %xmm0
  movsd %xmm0, 56(%rsp)
  movq  8(%rsp), %r10
  movsd %xmm0, (%r10, %rax, 8)

  movq  $0, 48(%rsp)
.Lfast_backward_input_loop:
  movq  48(%rsp), %rcx
  cmpq  %rbx, %rcx
  jae   .Lfast_backward_next_output
  movq  40(%rsp), %rax
  imulq %rbx, %rax
  addq  %rcx, %rax
  movsd 56(%rsp), %xmm0
  mulsd (%r12, %rcx, 8), %xmm0
  movq  0(%rsp), %r10
  movsd %xmm0, (%r10, %rax, 8)

  movsd (%r13, %rax, 8), %xmm0
  mulsd 56(%rsp), %xmm0
  movq  16(%rsp), %r10
  addsd (%r10, %rcx, 8), %xmm0
  movsd %xmm0, (%r10, %rcx, 8)
  incq  %rcx
  movq  %rcx, 48(%rsp)
  jmp   .Lfast_backward_input_loop

.Lfast_backward_next_output:
  movq  40(%rsp), %rax
  incq  %rax
  movq  %rax, 40(%rsp)
  jmp   .Lfast_backward_output_loop

.Lfast_backward_done:
  addq  $64, %rsp
  popq  %r15
  popq  %r14
  popq  %r13
  popq  %r12
  popq  %rbx
  ret

  .section .note.GNU-stack,"",@progbits
