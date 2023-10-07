.att_syntax prefix

#if defined(__USER_LABEL_PREFIX__)
#define CONCAT(x, y) x ## y
#define CONCATENATION(x, y) CONCAT(x, y)
#define add_mod_384 CONCATENATION(__USER_LABEL_PREFIX__, add_mod_384)
#endif

.text

.globl	add_mod_384
add_mod_384:
	push	%rbp
	push	%rbx
	push	%r12
	push	%r13
	push	%r14
	push	%r15
	sub	$8, %rsp

	call	__add_mod_384

	mov	8(%rsp),%r15
	mov	16(%rsp),%r14
	mov	24(%rsp),%r13
	mov	32(%rsp),%r12
	mov	40(%rsp),%rbx
	mov	48(%rsp),%rbp
	lea	56(%rsp),%rsp
	ret

__add_mod_384:
	mov	8*0(%rsi), %r8
	mov	8*1(%rsi), %r9
	mov	8*2(%rsi), %r10
	mov	8*3(%rsi), %r11
	mov	8*4(%rsi), %r12
	mov	8*5(%rsi), %r13

__add_mod_384_a_is_loaded:
	add	8*0(%rdx), %r8
	adc	8*1(%rdx), %r9
	adc	8*2(%rdx), %r10
	 mov	%r8, %r14
	adc	8*3(%rdx), %r11
	 mov	%r9, %r15
	adc	8*4(%rdx), %r12
	 mov	%r10, %rax
	adc	8*5(%rdx), %r13
	 mov	%r11, %rbx
	sbb	%rdx, %rdx

	sub	8*0(%rcx), %r8
	sbb	8*1(%rcx), %r9
	 mov	%r12, %rbp
	sbb	8*2(%rcx), %r10
	sbb	8*3(%rcx), %r11
	sbb	8*4(%rcx), %r12
	 mov	%r13, %rsi
	sbb	8*5(%rcx), %r13
	sbb	$0, %rdx

	cmovc	%r14,  %r8
	cmovc	%r15,  %r9
	cmovc	%rax,  %r10
	mov	%r8, 8*0(%rdi)
	cmovc	%rbx,  %r11
	mov	%r9, 8*1(%rdi)
	cmovc	%rbp, %r12
	mov	%r10, 8*2(%rdi)
	cmovc	%rsi, %r13
	mov	%r11, 8*3(%rdi)
	mov	%r12, 8*4(%rdi)
	mov	%r13, 8*5(%rdi)

	ret
