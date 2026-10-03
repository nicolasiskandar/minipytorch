  .text
  .globl dotProductAsm
  .type dotProductAsm, @function

dotProductAsm:
  xor   %rax, %rax
  pxor  %xmm0, %xmm0
  test  %rdx, %rdx
  je    .Ldone

.Lloop:
  movsd   (%rdi, %rax, 8), %xmm1
  mulsd   (%rsi, %rax, 8), %xmm1
  addsd   %xmm1, %xmm0
  inc     %rax
  cmp     %rdx, %rax
  jne     .Lloop

.Ldone:
  ret

  .section .note.GNU-stack,"",@progbits