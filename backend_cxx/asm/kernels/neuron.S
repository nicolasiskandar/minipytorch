  .text
  .globl nn_neuron_forward_f64
  .type nn_neuron_forward_f64, @function

# double nn_neuron_forward_f64(const double* input, const double* weights,
#                              size_t count, double bias, int activationKind)
# Arguments: rdi=input, rsi=weights, rdx=count, xmm0=bias, rcx=activation.
nn_neuron_forward_f64:
  pushq %rbx
  subq  $16, %rsp
  movq  %rcx, %rbx
  movsd %xmm0, 0(%rsp)
  xchgq %rdi, %rsi
  call  nn_dot_product_f64
  addsd 0(%rsp), %xmm0
  cmpq  $0, %rbx
  je    .Lneuron_sigmoid
  cmpq  $1, %rbx
  je    .Lneuron_tanh
  call  nn_relu_f64
  jmp   .Lneuron_done
.Lneuron_sigmoid:
  call  nn_sigmoid_f64
  jmp   .Lneuron_done
.Lneuron_tanh:
  call  nn_tanh_f64
.Lneuron_done:
  addq  $16, %rsp
  popq  %rbx
  ret

  .globl nn_neuron_backward_f64
  .type nn_neuron_backward_f64, @function

# The first six integer arguments are in rdi, rsi, rdx, rcx, r8, and r9;
# dLoss_dInput is the tenth argument and is passed on the stack.
nn_neuron_backward_f64:
  pushq %rbx
  pushq %r12
  pushq %r13
  subq  $64, %rsp
  movq  %rdi, %r12
  movq  %rsi, %r13
  movq  %rdx, %rbx
  movq  %r8, 0(%rsp)
  movq  %r9, 8(%rsp)
  movq  96(%rsp), %rax
  movq  %rax, 16(%rsp)
  movq  %rcx, 24(%rsp)
  movsd %xmm1, 32(%rsp)

  cmpq  $0, 24(%rsp)
  je    .Lneuron_backward_sigmoid
  cmpq  $1, 24(%rsp)
  je    .Lneuron_backward_tanh
  call  nn_relu_derivative_from_output_f64
  jmp   .Lneuron_backward_delta
.Lneuron_backward_sigmoid:
  call  nn_sigmoid_derivative_from_output_f64
  jmp   .Lneuron_backward_delta
.Lneuron_backward_tanh:
  call  nn_tanh_derivative_from_output_f64
.Lneuron_backward_delta:
  mulsd 32(%rsp), %xmm0
  movsd %xmm0, 40(%rsp)
  movq  8(%rsp), %r10
  movsd %xmm0, (%r10)

  movq  $0, 48(%rsp)
.Lneuron_backward_loop:
  movq  48(%rsp), %rax
  cmpq  %rbx, %rax
  jae   .Lneuron_backward_done
  movsd 40(%rsp), %xmm0
  mulsd (%r13, %rax, 8), %xmm0
  movq  0(%rsp), %r10
  movsd %xmm0, (%r10, %rax, 8)

  movsd (%r12, %rax, 8), %xmm0
  mulsd 40(%rsp), %xmm0
  movq  16(%rsp), %r10
  movsd %xmm0, (%r10, %rax, 8)
  incq  %rax
  movq  %rax, 48(%rsp)
  jmp   .Lneuron_backward_loop
.Lneuron_backward_done:
  addq  $64, %rsp
  popq  %r13
  popq  %r12
  popq  %rbx
  ret

  .section .note.GNU-stack,"",@progbits
