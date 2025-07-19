
#if defined(__USER_LABEL_PREFIX__)
#if !(~(~__USER_LABEL_PREFIX__ + 0) == 0 && ~(~__USER_LABEL_PREFIX__ + 1) == 1)
#define CONCAT(x, y) x ## y
#define CONCATENATION(x, y) CONCAT(x, y)
#define read_tsc CONCATENATION(__USER_LABEL_PREFIX__, read_tsc)
#endif
#endif


.intel_syntax noprefix      // we use Intel syntax
.text                       // place current section in code segment


.global read_tsc            // make <symbol> visible outside this module

#if defined(__ELF__)
.type read_tsc, @function   // declare <symbol> to be a function
.func read_tsc
#endif


read_tsc:
    push    rbx             // ebx gets modified by cpuid
    xor     eax, eax        // set eax to 0
    cpuid                   // serialize instruction execution
    rdtsc                   // read out TSC and place it in edx:eax
    shl     rdx, 32         // edx holds the upper 32 bits of 64-bit TSC value
    or      rax, rdx        // combine upper & lower 32 bits in 64-bit reg rax
    push    rax             // push rax on stack
    xor     eax, eax        // set eax to 0
    cpuid                   // serialize instruction execution
    pop     rax             // restore rax (holds return value)
    pop     rbx             // restore rbx
    ret                     // return to caller

#if defined(__ELF__)
.endfunc
.size read_tsc, .-read_tsc
#endif


.att_syntax prefix          // switch back to AT&T syntax


.end
