$$
y_i|\mu_i \sim (\mu_i, \sigma^2)
$$

$$
\text{df} = \frac{1}{\sigma^2}\sum_{i=1}^n\text{Cov}(y_i, \hat{y}_i)=\frac{1}{\sigma^2}\sum_{i=1}^n\mathrm{E}[(y_i-\mu_i)(\hat{y}_i-\mu_i)]=\frac{1}{\sigma^2}\sum_{i=1}^n\mathrm{E}[(y_i-\mu_i)e_i]\\ \sim \frac{1}{\sigma^2}\sum_{i=1}^n[e_i(\hat{y}_i-\mu_i)]\sim \frac{1}{\sigma^2}\sum_{i=1}^ne_i\hat{y}_i
$$

Simulation df:

1. Generate data: $Y_0 = f(X)$
2. Simulation $nrep$ times. For each time,
   + Simulatie $e \sim N(0, \sigma^2I)$ and $Y = f(X)+e$
   + Model fit $\hat{Y}$
   + Evaluate $\text{df} = \frac{1}{\sigma^2}\sum_{i=1}^n\hat{Y}_i e_i$

3. Take average of all the $\text{df}$ simulated.