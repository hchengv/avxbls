# Analysis wrt. final expo

## With cyclotomic decomposition

The formula for the hard part of the final exponentiation to be implemented is given in https://eprint.iacr.org/2020/875.pdf, page 14. 

$$ 3 \cdot \frac{\Phi_{12} (p(z))}{r(z)} = (z - 1)^2 \cdot (z + p) \cdot (z^2 + p^2 - 1) + 3 $$

where $z$ is the BLS12-381 seed, namely: $z = \texttt{0xd201000000010000}$. 

We use the following table to convert operations in extention fields to operations over the base field $\mathbb F_p$.

k|1|12
---|:---:|---: 
$m_k$                 |1**m**  |54**m**
$s_k$                 |1**m**  |36**m**
$f_k$                 |0       |10**m**
$s_k^{\text{cyclo}}$  |-       |18**m**
$i_k$                 |25**m** |119**m**

The exponentiation by an integer $z$ is computed as follows: 

$$ 1 exp_z = (\text{bitsize}(z) - 1) s_{12}^{\text{cyclo}} + (\text{hw}(z) - 1) m_{12} $$

### The BLST implementation

```r
// BLST final exp. - hard part
y0 <- ret^2     = f^2                                                          // 1 s_12^cyclo
y1 <- y0^z      = f^(2*z)                                                      // 1 exp_z
y2 <- y1^(z/2)  = f^(z^2)                                                      // 1 exp_z/2
y3 <- ret       = f                                                        
y3 <- y3^(p^6)  = f^(-1)                                                       // 1 conj
y1 <- y1*y3     = f^(2*z - 1)                                                  // 1 m_12
y1 <- y1^(p^6)  = f^(1 - 2*z)                                                  // 1 conj  
y1 <- y1*y2     = f^(1 - z)^2                                                  // 1 m_12
y2 <- y1^z      = f^[z*(1 - z)^2]                                              // 1 exp_z
y3 <- y2^z      = f^[z^2*(1 - z)^2]                                            // 1 exp_z
y1 <- y1^(p^6)  = f^[-(1 - z)^2]                                               // 1 conj      
y3 <- y3*y1     = f^[(z^2 - 1)*(1 - z)^2]                                      // 1 m_12
y1 <- y1^(p^6)  = f^(1 - z)^2                                                  // 1 conj  
y1 <- y1^(p^3)  = f^[p^3*(1 - z)^2]                                            // 1 f
y2 <- y2^(p^2)  = f^[z*(1 - z)^2*p^2]                                          // 1 f
y1 <- y1*y2     = f^[(z + p)*(1 - z)^2*p^2]                                    // 1 m_12
y2 <- y3^z      = f^[z*(z^2 - 1)*(1 - z)^2]                                    // 1 exp_z
y2 <- y2*y0     = f^[z*(z^2 - 1)*(1 - z)^2 + 2]                                // 1 m_12
y2 <- y2*ret    = f^[z*(z^2 - 1)*(1 - z)^2 + 3]                                // 1 m_12
y1 <- y1*y2     = f^[(z + p)*(1 - z)^2*p^2 + z*(z^2 - 1)*(1 - z)^2 + 3]        // 1 m_12
y2 <- y3^p      = f^[p*(z^2 - 1)*(1 - z)^2]                                    // 1 f
ret <- y1*y2    = f^[(1 - z)^2*(z + p)*(z^2 + p^2 - 1) + 3]                    // 1 m_12
```
This version of the hard part of the final exponentiation requires: 

$$ 8 m_{12} + 1 s_2^{\text{cyclo}} + 1 exp_{z/2} + 4exp_z + 3f + 4 \text{conj} = 7482 m$$ 

### The SageMath implementation

```r
// SageMath final exp. - hard part
m1 <- ret^(z - 1)  = f^(z - 1)                                                 // 1 exp_(z - 1)
m2 <- m1^(z - 1)   = f^[(z - 1)^2]                                             // 1 exp_(z - 1)
m3 <- m2^z         = f^[z*(z - 1)^2]                                           // 1 exp_z
m4 <- m2^p         = f^[p*(z - 1)^2]                                           // 1 f
m5 <- m4*m3        = f^[(p + u)*(z - 1)^2]                                     // 1 m_12           
m6 <- m5^(p^2)     = f^[p^2*(p + u)*(z - 1)^2]                                 // 1 f
m7 <- m5^z         = f^[z*(p + u)*(z - 1)^2]                                   // 1 exp_z
m8 <- m7^z         = f^[z^2*(p + u)*(z - 1)^2]                                 // 1 exp_z
m9 <- m5^(p^6)     = f^[-(p + u)*(z - 1)^2]                                    // 1 f
m10 <- m6*m8       = f^[p^2*(p + u)*(z - 1)^2 + z^2*(p + u)*(z - 1)^2]         // 1 m_12
m11 <- m10*m9      = f^[(p + u)*(z - 1)^2*(z^2 + p^2 - 1)]                     // 1 m_12
m12 <- ret^2       = f^2                                                       // 1 s_12^cyclo
m13 <- m12*ret     = f^3                                                       // 1 m_12
m14 <- m11*m13     = f^[(p + u)*(z - 1)^2*(z^2 + p^2 - 1) + 3]                 // 1 m_12
```

This version of the hard part of the final exponentiation requires: 

$$ 5 m_{12} + 1 s_2^{\text{cyclo}} + 2 exp_{z-1} + 3exp_z + 2f + 1 \text{conj} = 7436 m$$ 

## With Lattice Reduction
