
.intel_syntax noprefix


#if defined(__USER_LABEL_PREFIX__)
#define CONCAT(x, y) x ## y
#define CONCATENATION(x, y) CONCAT(x, y)
#define add_mod_384 CONCATENATION(__USER_LABEL_PREFIX__, add_mod_384)
#define add_mod_384x CONCATENATION(__USER_LABEL_PREFIX__, add_mod_384x)
#define lshift_mod_384 CONCATENATION(__USER_LABEL_PREFIX__, lshift_mod_384)
#define mul_by_3_mod_384 CONCATENATION(__USER_LABEL_PREFIX__, mul_by_3_mod_384)
#define mul_by_8_mod_384 CONCATENATION(__USER_LABEL_PREFIX__, mul_by_8_mod_384)
#endif


($r_ptr,$a_ptr,$b_org,$n_ptr,$n0) = ("%rdi","%rsi","%rdx","%rcx","%r8");
$b_ptr = "%rbx";

my @acc=map("%r$_",(8..15, "ax", "bx", "bp"));
   push(@acc, $a_ptr);

$code.=<<___;

.text

.globl	add_mod_384
.hidden	add_mod_384
.type	add_mod_384,\@function,4,"unwind"
.align	32
add_mod_384:
.cfi_startproc
	push	%rbp
.cfi_push	%rbp
	push	%rbx
.cfi_push	%rbx
	push	%r12
.cfi_push	%r12
	push	%r13
.cfi_push	%r13
	push	%r14
.cfi_push	%r14
	push	%r15
.cfi_push	%r15
	sub	\$8, %rsp
.cfi_adjust_cfa_offset	8
.cfi_end_prologue

	call	__add_mod_384

	mov	8(%rsp),%r15
.cfi_restore	%r15
	mov	16(%rsp),%r14
.cfi_restore	%r14
	mov	24(%rsp),%r13
.cfi_restore	%r13
	mov	32(%rsp),%r12
.cfi_restore	%r12
	mov	40(%rsp),%rbx
.cfi_restore	%rbx
	mov	48(%rsp),%rbp
.cfi_restore	%rbp
	lea	56(%rsp),%rsp
.cfi_adjust_cfa_offset	-56
.cfi_epilogue
	ret
.cfi_endproc
.size	add_mod_384,.-add_mod_384

.type	__add_mod_384,\@abi-omnipotent
.align	32
__add_mod_384:
	mov	8*0($a_ptr), @acc[0]
	mov	8*1($a_ptr), @acc[1]
	mov	8*2($a_ptr), @acc[2]
	mov	8*3($a_ptr), @acc[3]
	mov	8*4($a_ptr), @acc[4]
	mov	8*5($a_ptr), @acc[5]

__add_mod_384_a_is_loaded:
	add	8*0($b_org), @acc[0]
	adc	8*1($b_org), @acc[1]
	adc	8*2($b_org), @acc[2]
	 mov	@acc[0], @acc[6]
	adc	8*3($b_org), @acc[3]
	 mov	@acc[1], @acc[7]
	adc	8*4($b_org), @acc[4]
	 mov	@acc[2], @acc[8]
	adc	8*5($b_org), @acc[5]
	 mov	@acc[3], @acc[9]
	sbb	$b_org, $b_org

	sub	8*0($n_ptr), @acc[0]
	sbb	8*1($n_ptr), @acc[1]
	 mov	@acc[4], @acc[10]
	sbb	8*2($n_ptr), @acc[2]
	sbb	8*3($n_ptr), @acc[3]
	sbb	8*4($n_ptr), @acc[4]
	 mov	@acc[5], @acc[11]
	sbb	8*5($n_ptr), @acc[5]
	sbb	\$0, $b_org

	cmovc	@acc[6],  @acc[0]
	cmovc	@acc[7],  @acc[1]
	cmovc	@acc[8],  @acc[2]
	mov	@acc[0], 8*0($r_ptr)
	cmovc	@acc[9],  @acc[3]
	mov	@acc[1], 8*1($r_ptr)
	cmovc	@acc[10], @acc[4]
	mov	@acc[2], 8*2($r_ptr)
	cmovc	@acc[11], @acc[5]
	mov	@acc[3], 8*3($r_ptr)
	mov	@acc[4], 8*4($r_ptr)
	mov	@acc[5], 8*5($r_ptr)

	ret
.size	__add_mod_384,.-__add_mod_384

.globl	add_mod_384x
.hidden	add_mod_384x
.type	add_mod_384x,\@function,4,"unwind"
.align	32
add_mod_384x:
.cfi_startproc
	push	%rbp
.cfi_push	%rbp
	push	%rbx
.cfi_push	%rbx
	push	%r12
.cfi_push	%r12
	push	%r13
.cfi_push	%r13
	push	%r14
.cfi_push	%r14
	push	%r15
.cfi_push	%r15
	sub	\$24, %rsp
.cfi_adjust_cfa_offset	24
.cfi_end_prologue

	mov	$a_ptr, 8*0(%rsp)
	mov	$b_org, 8*1(%rsp)
	lea	48($a_ptr), $a_ptr	# a->im
	lea	48($b_org), $b_org	# b->im
	lea	48($r_ptr), $r_ptr	# ret->im
	call	__add_mod_384		# add_mod_384(ret->im, a->im, b->im, mod);

	mov	8*0(%rsp), $a_ptr	# a->re
	mov	8*1(%rsp), $b_org	# b->re
	lea	-48($r_ptr), $r_ptr	# ret->re
	call	__add_mod_384		# add_mod_384(ret->re, a->re, b->re, mod);

	mov	24+8*0(%rsp),%r15
.cfi_restore	%r15
	mov	24+8*1(%rsp),%r14
.cfi_restore	%r14
	mov	24+8*2(%rsp),%r13
.cfi_restore	%r13
	mov	24+8*3(%rsp),%r12
.cfi_restore	%r12
	mov	24+8*4(%rsp),%rbx
.cfi_restore	%rbx
	mov	24+8*5(%rsp),%rbp
.cfi_restore	%rbp
	lea	24+8*6(%rsp),%rsp
.cfi_adjust_cfa_offset	-24-8*6
.cfi_epilogue
	ret
.cfi_endproc
.size	add_mod_384x,.-add_mod_384x

########################################################################
.globl	lshift_mod_384
.hidden	lshift_mod_384
.type	lshift_mod_384,\@function,4,"unwind"
.align	32
lshift_mod_384:
.cfi_startproc
	push	%rbp
.cfi_push	%rbp
	push	%rbx
.cfi_push	%rbx
	push	%r12
.cfi_push	%r12
	push	%r13
.cfi_push	%r13
	push	%r14
.cfi_push	%r14
	push	%r15
.cfi_push	%r15
	push	$r_ptr
.cfi_adjust_cfa_offset	8
.cfi_end_prologue

	mov	8*0($a_ptr), @acc[0]
	mov	8*1($a_ptr), @acc[1]
	mov	8*2($a_ptr), @acc[2]
	mov	8*3($a_ptr), @acc[3]
	mov	8*4($a_ptr), @acc[4]
	mov	8*5($a_ptr), @acc[5]

.Loop_lshift_mod_384:
	add	@acc[0], @acc[0]
	adc	@acc[1], @acc[1]
	adc	@acc[2], @acc[2]
	 mov	@acc[0], @acc[6]
	adc	@acc[3], @acc[3]
	 mov	@acc[1], @acc[7]
	adc	@acc[4], @acc[4]
	 mov	@acc[2], @acc[8]
	adc	@acc[5], @acc[5]
	 mov	@acc[3], @acc[9]
	sbb	$r_ptr, $r_ptr

	sub	8*0($n_ptr), @acc[0]
	sbb	8*1($n_ptr), @acc[1]
	 mov	@acc[4], @acc[10]
	sbb	8*2($n_ptr), @acc[2]
	sbb	8*3($n_ptr), @acc[3]
	sbb	8*4($n_ptr), @acc[4]
	 mov	@acc[5], @acc[11]
	sbb	8*5($n_ptr), @acc[5]
	sbb	\$0, $r_ptr

	mov	(%rsp), $r_ptr
	cmovc	@acc[6],  @acc[0]
	cmovc	@acc[7],  @acc[1]
	cmovc	@acc[8],  @acc[2]
	cmovc	@acc[9],  @acc[3]
	cmovc	@acc[10], @acc[4]
	cmovc	@acc[11], @acc[5]

	dec	%edx
	jnz	.Loop_lshift_mod_384

	mov	@acc[0], 8*0($r_ptr)
	mov	@acc[1], 8*1($r_ptr)
	mov	@acc[2], 8*2($r_ptr)
	mov	@acc[3], 8*3($r_ptr)
	mov	@acc[4], 8*4($r_ptr)
	mov	@acc[5], 8*5($r_ptr)

	mov	8(%rsp),%r15
.cfi_restore	%r15
	mov	16(%rsp),%r14
.cfi_restore	%r14
	mov	24(%rsp),%r13
.cfi_restore	%r13
	mov	32(%rsp),%r12
.cfi_restore	%r12
	mov	40(%rsp),%rbx
.cfi_restore	%rbx
	mov	48(%rsp),%rbp
.cfi_restore	%rbp
	lea	56(%rsp),%rsp
.cfi_adjust_cfa_offset	-56
.cfi_epilogue
	ret
.cfi_endproc
.size	lshift_mod_384,.-lshift_mod_384

.type	__lshift_mod_384,\@abi-omnipotent
.align	32
__lshift_mod_384:
	add	@acc[0], @acc[0]
	adc	@acc[1], @acc[1]
	adc	@acc[2], @acc[2]
	 mov	@acc[0], @acc[6]
	adc	@acc[3], @acc[3]
	 mov	@acc[1], @acc[7]
	adc	@acc[4], @acc[4]
	 mov	@acc[2], @acc[8]
	adc	@acc[5], @acc[5]
	 mov	@acc[3], @acc[9]
	sbb	$b_org, $b_org

	sub	8*0($n_ptr), @acc[0]
	sbb	8*1($n_ptr), @acc[1]
	 mov	@acc[4], @acc[10]
	sbb	8*2($n_ptr), @acc[2]
	sbb	8*3($n_ptr), @acc[3]
	sbb	8*4($n_ptr), @acc[4]
	 mov	@acc[5], @acc[11]
	sbb	8*5($n_ptr), @acc[5]
	sbb	\$0, $b_org

	cmovc	@acc[6],  @acc[0]
	cmovc	@acc[7],  @acc[1]
	cmovc	@acc[8],  @acc[2]
	cmovc	@acc[9],  @acc[3]
	cmovc	@acc[10], @acc[4]
	cmovc	@acc[11], @acc[5]

	ret
.size	__lshift_mod_384,.-__lshift_mod_384

########################################################################
.globl	mul_by_3_mod_384
.hidden	mul_by_3_mod_384
.type	mul_by_3_mod_384,\@function,3,"unwind"
.align	32
mul_by_3_mod_384:
.cfi_startproc
	push	%rbp
.cfi_push	%rbp
	push	%rbx
.cfi_push	%rbx
	push	%r12
.cfi_push	%r12
	push	%r13
.cfi_push	%r13
	push	%r14
.cfi_push	%r14
	push	%r15
.cfi_push	%r15
	push	$a_ptr
.cfi_adjust_cfa_offset	8
.cfi_end_prologue

	mov	8*0($a_ptr), @acc[0]
	mov	8*1($a_ptr), @acc[1]
	mov	8*2($a_ptr), @acc[2]
	mov	8*3($a_ptr), @acc[3]
	mov	8*4($a_ptr), @acc[4]
	mov	8*5($a_ptr), @acc[5]
	mov	$b_org, $n_ptr

	call	__lshift_mod_384

	mov	(%rsp), $b_org
	call	__add_mod_384_a_is_loaded

	mov	8(%rsp),%r15
.cfi_restore	%r15
	mov	16(%rsp),%r14
.cfi_restore	%r14
	mov	24(%rsp),%r13
.cfi_restore	%r13
	mov	32(%rsp),%r12
.cfi_restore	%r12
	mov	40(%rsp),%rbx
.cfi_restore	%rbx
	mov	48(%rsp),%rbp
.cfi_restore	%rbp
	lea	56(%rsp),%rsp
.cfi_adjust_cfa_offset	-56
.cfi_epilogue
	ret
.cfi_endproc
.size	mul_by_3_mod_384,.-mul_by_3_mod_384

.globl	mul_by_8_mod_384
.hidden	mul_by_8_mod_384
.type	mul_by_8_mod_384,\@function,3,"unwind"
.align	32
mul_by_8_mod_384:
.cfi_startproc
	push	%rbp
.cfi_push	%rbp
	push	%rbx
.cfi_push	%rbx
	push	%r12
.cfi_push	%r12
	push	%r13
.cfi_push	%r13
	push	%r14
.cfi_push	%r14
	push	%r15
.cfi_push	%r15
	sub	\$8, %rsp
.cfi_adjust_cfa_offset	8
.cfi_end_prologue

	mov	8*0($a_ptr), @acc[0]
	mov	8*1($a_ptr), @acc[1]
	mov	8*2($a_ptr), @acc[2]
	mov	8*3($a_ptr), @acc[3]
	mov	8*4($a_ptr), @acc[4]
	mov	8*5($a_ptr), @acc[5]
	mov	$b_org, $n_ptr

	call	__lshift_mod_384
	call	__lshift_mod_384
	call	__lshift_mod_384

	mov	@acc[0], 8*0($r_ptr)
	mov	@acc[1], 8*1($r_ptr)
	mov	@acc[2], 8*2($r_ptr)
	mov	@acc[3], 8*3($r_ptr)
	mov	@acc[4], 8*4($r_ptr)
	mov	@acc[5], 8*5($r_ptr)

	mov	8(%rsp),%r15
.cfi_restore	%r15
	mov	16(%rsp),%r14
.cfi_restore	%r14
	mov	24(%rsp),%r13
.cfi_restore	%r13
	mov	32(%rsp),%r12
.cfi_restore	%r12
	mov	40(%rsp),%rbx
.cfi_restore	%rbx
	mov	48(%rsp),%rbp
.cfi_restore	%rbp
	lea	56(%rsp),%rsp
.cfi_adjust_cfa_offset	-56
.cfi_epilogue
	ret
.cfi_endproc
.size	mul_by_8_mod_384,.-mul_by_8_mod_384

########################################################################
.globl	mul_by_3_mod_384x
.hidden	mul_by_3_mod_384x
.type	mul_by_3_mod_384x,\@function,3,"unwind"
.align	32
mul_by_3_mod_384x:
.cfi_startproc
	push	%rbp
.cfi_push	%rbp
	push	%rbx
.cfi_push	%rbx
	push	%r12
.cfi_push	%r12
	push	%r13
.cfi_push	%r13
	push	%r14
.cfi_push	%r14
	push	%r15
.cfi_push	%r15
	push	$a_ptr
.cfi_adjust_cfa_offset	8
.cfi_end_prologue

	mov	8*0($a_ptr), @acc[0]
	mov	8*1($a_ptr), @acc[1]
	mov	8*2($a_ptr), @acc[2]
	mov	8*3($a_ptr), @acc[3]
	mov	8*4($a_ptr), @acc[4]
	mov	8*5($a_ptr), @acc[5]
	mov	$b_org, $n_ptr

	call	__lshift_mod_384

	mov	(%rsp), $b_org
	call	__add_mod_384_a_is_loaded

	mov	(%rsp), $a_ptr
	lea	8*6($r_ptr), $r_ptr

	mov	8*6($a_ptr), @acc[0]
	mov	8*7($a_ptr), @acc[1]
	mov	8*8($a_ptr), @acc[2]
	mov	8*9($a_ptr), @acc[3]
	mov	8*10($a_ptr), @acc[4]
	mov	8*11($a_ptr), @acc[5]

	call	__lshift_mod_384

	mov	\$8*6, $b_org
	add	(%rsp), $b_org
	call	__add_mod_384_a_is_loaded

	mov	8(%rsp),%r15
.cfi_restore	%r15
	mov	16(%rsp),%r14
.cfi_restore	%r14
	mov	24(%rsp),%r13
.cfi_restore	%r13
	mov	32(%rsp),%r12
.cfi_restore	%r12
	mov	40(%rsp),%rbx
.cfi_restore	%rbx
	mov	48(%rsp),%rbp
.cfi_restore	%rbp
	lea	56(%rsp),%rsp
.cfi_adjust_cfa_offset	-56
.cfi_epilogue
	ret
.cfi_endproc
.size	mul_by_3_mod_384x,.-mul_by_3_mod_384x

.globl	mul_by_8_mod_384x
.hidden	mul_by_8_mod_384x
.type	mul_by_8_mod_384x,\@function,3,"unwind"
.align	32
mul_by_8_mod_384x:
.cfi_startproc
	push	%rbp
.cfi_push	%rbp
	push	%rbx
.cfi_push	%rbx
	push	%r12
.cfi_push	%r12
	push	%r13
.cfi_push	%r13
	push	%r14
.cfi_push	%r14
	push	%r15
.cfi_push	%r15
	push	$a_ptr
.cfi_adjust_cfa_offset	8
.cfi_end_prologue

	mov	8*0($a_ptr), @acc[0]
	mov	8*1($a_ptr), @acc[1]
	mov	8*2($a_ptr), @acc[2]
	mov	8*3($a_ptr), @acc[3]
	mov	8*4($a_ptr), @acc[4]
	mov	8*5($a_ptr), @acc[5]
	mov	$b_org, $n_ptr

	call	__lshift_mod_384
	call	__lshift_mod_384
	call	__lshift_mod_384

	mov	(%rsp), $a_ptr
	mov	@acc[0], 8*0($r_ptr)
	mov	@acc[1], 8*1($r_ptr)
	mov	@acc[2], 8*2($r_ptr)
	mov	@acc[3], 8*3($r_ptr)
	mov	@acc[4], 8*4($r_ptr)
	mov	@acc[5], 8*5($r_ptr)

	mov	48+8*0($a_ptr), @acc[0]
	mov	48+8*1($a_ptr), @acc[1]
	mov	48+8*2($a_ptr), @acc[2]
	mov	48+8*3($a_ptr), @acc[3]
	mov	48+8*4($a_ptr), @acc[4]
	mov	48+8*5($a_ptr), @acc[5]

	call	__lshift_mod_384
	call	__lshift_mod_384
	call	__lshift_mod_384

	mov	@acc[0], 48+8*0($r_ptr)
	mov	@acc[1], 48+8*1($r_ptr)
	mov	@acc[2], 48+8*2($r_ptr)
	mov	@acc[3], 48+8*3($r_ptr)
	mov	@acc[4], 48+8*4($r_ptr)
	mov	@acc[5], 48+8*5($r_ptr)

	mov	8(%rsp),%r15
.cfi_restore	%r15
	mov	16(%rsp),%r14
.cfi_restore	%r14
	mov	24(%rsp),%r13
.cfi_restore	%r13
	mov	32(%rsp),%r12
.cfi_restore	%r12
	mov	40(%rsp),%rbx
.cfi_restore	%rbx
	mov	48(%rsp),%rbp
.cfi_restore	%rbp
	lea	56(%rsp),%rsp
.cfi_adjust_cfa_offset	-56
.cfi_epilogue
	ret
.cfi_endproc
.size	mul_by_8_mod_384x,.-mul_by_8_mod_384x

########################################################################
.globl	cneg_mod_384
.hidden	cneg_mod_384
.type	cneg_mod_384,\@function,4,"unwind"
.align	32
cneg_mod_384:
.cfi_startproc
	push	%rbp
.cfi_push	%rbp
	push	%rbx
.cfi_push	%rbx
	push	%r12
.cfi_push	%r12
	push	%r13
.cfi_push	%r13
	push	%r14
.cfi_push	%r14
	push	%r15
.cfi_push	%r15
	push	$b_org			# condition flag
.cfi_adjust_cfa_offset	8
.cfi_end_prologue

	mov	8*0($a_ptr), $b_org	# load a[0:5]
	mov	8*1($a_ptr), @acc[1]
	mov	8*2($a_ptr), @acc[2]
	mov	$b_org, @acc[0]
	mov	8*3($a_ptr), @acc[3]
	or	@acc[1], $b_org
	mov	8*4($a_ptr), @acc[4]
	or	@acc[2], $b_org
	mov	8*5($a_ptr), @acc[5]
	or	@acc[3], $b_org
	mov	\$-1, @acc[11]
	or	@acc[4], $b_org
	or	@acc[5], $b_org

	mov	8*0($n_ptr), @acc[6]	# load n[0:5]
	cmovnz	@acc[11], $b_org	# mask = a[0:5] ? -1 : 0
	mov	8*1($n_ptr), @acc[7]
	mov	8*2($n_ptr), @acc[8]
	and	$b_org, @acc[6]		# n[0:5] &= mask
	mov	8*3($n_ptr), @acc[9]
	and	$b_org, @acc[7]
	mov	8*4($n_ptr), @acc[10]
	and	$b_org, @acc[8]
	mov	8*5($n_ptr), @acc[11]
	and	$b_org, @acc[9]
	mov	0(%rsp), $n_ptr		# restore condition flag
	and	$b_org, @acc[10]
	and	$b_org, @acc[11]

	sub	@acc[0], @acc[6]	# a[0:5] ? n[0:5]-a[0:5] : 0-0
	sbb	@acc[1], @acc[7]
	sbb	@acc[2], @acc[8]
	sbb	@acc[3], @acc[9]
	sbb	@acc[4], @acc[10]
	sbb	@acc[5], @acc[11]

	or	$n_ptr, $n_ptr		# check condition flag

	cmovz	@acc[0], @acc[6]	# flag ? n[0:5]-a[0:5] : a[0:5]
	cmovz	@acc[1], @acc[7]
	cmovz	@acc[2], @acc[8]
	mov	@acc[6], 8*0($r_ptr)
	cmovz	@acc[3], @acc[9]
	mov	@acc[7], 8*1($r_ptr)
	cmovz	@acc[4], @acc[10]
	mov	@acc[8], 8*2($r_ptr)
	cmovz	@acc[5], @acc[11]
	mov	@acc[9], 8*3($r_ptr)
	mov	@acc[10], 8*4($r_ptr)
	mov	@acc[11], 8*5($r_ptr)

	mov	8(%rsp),%r15
.cfi_restore	%r15
	mov	16(%rsp),%r14
.cfi_restore	%r14
	mov	24(%rsp),%r13
.cfi_restore	%r13
	mov	32(%rsp),%r12
.cfi_restore	%r12
	mov	40(%rsp),%rbx
.cfi_restore	%rbx
	mov	48(%rsp),%rbp
.cfi_restore	%rbp
	lea	56(%rsp),%rsp
.cfi_adjust_cfa_offset	-56
.cfi_epilogue
	ret
.cfi_endproc
.size	cneg_mod_384,.-cneg_mod_384

########################################################################
.globl	sub_mod_384
.hidden	sub_mod_384
.type	sub_mod_384,\@function,4,"unwind"
.align	32
sub_mod_384:
.cfi_startproc
	push	%rbp
.cfi_push	%rbp
	push	%rbx
.cfi_push	%rbx
	push	%r12
.cfi_push	%r12
	push	%r13
.cfi_push	%r13
	push	%r14
.cfi_push	%r14
	push	%r15
.cfi_push	%r15
	sub	\$8, %rsp
.cfi_adjust_cfa_offset	8
.cfi_end_prologue

	call	__sub_mod_384

	mov	8(%rsp),%r15
.cfi_restore	%r15
	mov	16(%rsp),%r14
.cfi_restore	%r14
	mov	24(%rsp),%r13
.cfi_restore	%r13
	mov	32(%rsp),%r12
.cfi_restore	%r12
	mov	40(%rsp),%rbx
.cfi_restore	%rbx
	mov	48(%rsp),%rbp
.cfi_restore	%rbp
	lea	56(%rsp),%rsp
.cfi_adjust_cfa_offset	-56
.cfi_epilogue
	ret
.cfi_endproc
.size	sub_mod_384,.-sub_mod_384

.type	__sub_mod_384,\@abi-omnipotent
.align	32
__sub_mod_384:
	mov	8*0($a_ptr), @acc[0]
	mov	8*1($a_ptr), @acc[1]
	mov	8*2($a_ptr), @acc[2]
	mov	8*3($a_ptr), @acc[3]
	mov	8*4($a_ptr), @acc[4]
	mov	8*5($a_ptr), @acc[5]

	sub	8*0($b_org), @acc[0]
	 mov	8*0($n_ptr), @acc[6]
	sbb	8*1($b_org), @acc[1]
	 mov	8*1($n_ptr), @acc[7]
	sbb	8*2($b_org), @acc[2]
	 mov	8*2($n_ptr), @acc[8]
	sbb	8*3($b_org), @acc[3]
	 mov	8*3($n_ptr), @acc[9]
	sbb	8*4($b_org), @acc[4]
	 mov	8*4($n_ptr), @acc[10]
	sbb	8*5($b_org), @acc[5]
	 mov	8*5($n_ptr), @acc[11]
	sbb	$b_org, $b_org

	and	$b_org, @acc[6]
	and	$b_org, @acc[7]
	and	$b_org, @acc[8]
	and	$b_org, @acc[9]
	and	$b_org, @acc[10]
	and	$b_org, @acc[11]

	add	@acc[6], @acc[0]
	adc	@acc[7], @acc[1]
	mov	@acc[0], 8*0($r_ptr)
	adc	@acc[8], @acc[2]
	mov	@acc[1], 8*1($r_ptr)
	adc	@acc[9], @acc[3]
	mov	@acc[2], 8*2($r_ptr)
	adc	@acc[10], @acc[4]
	mov	@acc[3], 8*3($r_ptr)
	adc	@acc[11], @acc[5]
	mov	@acc[4], 8*4($r_ptr)
	mov	@acc[5], 8*5($r_ptr)

	ret
.size	__sub_mod_384,.-__sub_mod_384

.globl	sub_mod_384x
.hidden	sub_mod_384x
.type	sub_mod_384x,\@function,4,"unwind"
.align	32
sub_mod_384x:
.cfi_startproc
	push	%rbp
.cfi_push	%rbp
	push	%rbx
.cfi_push	%rbx
	push	%r12
.cfi_push	%r12
	push	%r13
.cfi_push	%r13
	push	%r14
.cfi_push	%r14
	push	%r15
.cfi_push	%r15
	sub	\$24, %rsp
.cfi_adjust_cfa_offset	24
.cfi_end_prologue

	mov	$a_ptr, 8*0(%rsp)
	mov	$b_org, 8*1(%rsp)
	lea	48($a_ptr), $a_ptr	# a->im
	lea	48($b_org), $b_org	# b->im
	lea	48($r_ptr), $r_ptr	# ret->im
	call	__sub_mod_384		# sub_mod_384(ret->im, a->im, b->im, mod);

	mov	8*0(%rsp), $a_ptr	# a->re
	mov	8*1(%rsp), $b_org	# b->re
	lea	-48($r_ptr), $r_ptr	# ret->re
	call	__sub_mod_384		# sub_mod_384(ret->re, a->re, b->re, mod);

	mov	24+8*0(%rsp),%r15
.cfi_restore	%r15
	mov	24+8*1(%rsp),%r14
.cfi_restore	%r14
	mov	24+8*2(%rsp),%r13
.cfi_restore	%r13
	mov	24+8*3(%rsp),%r12
.cfi_restore	%r12
	mov	24+8*4(%rsp),%rbx
.cfi_restore	%rbx
	mov	24+8*5(%rsp),%rbp
.cfi_restore	%rbp
	lea	24+8*6(%rsp),%rsp
.cfi_adjust_cfa_offset	-24-8*6
.cfi_epilogue
	ret
.cfi_endproc
.size	sub_mod_384x,.-sub_mod_384x