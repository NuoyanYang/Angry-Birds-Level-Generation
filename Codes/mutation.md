### 1 step size:
*   $\sigma'\leftarrow\sigma\cdot e^{\tau \cdot N(0,1)}$
*   $\sigma'\leftarrow\max(\sigma',\varepsilon_0)$
*   $x_i'\leftarrow x_i+\sigma'\cdot N_i(0,1)$
*   $\tau\propto1/\sqrt{n}$

<br>

### n step size:
*   $\sigma_i'\leftarrow\sigma_i\cdot e^{\tau' \cdot N(0,1)+\tau \cdot N_i(0,1)}$
    *   or $\sigma_i'\leftarrow\sigma_i\cdot e^{\tau \cdot N_i(0,1)}$
*   $\sigma_i'\leftarrow\max(\sigma_i',\varepsilon_0)$
*   $x_i'\leftarrow x_i+\sigma_i'\cdot N_i(0,1)$
*   $\tau'\propto1/\sqrt{2n}$
*   $\tau\propto1/\sqrt{2\sqrt{n}}$