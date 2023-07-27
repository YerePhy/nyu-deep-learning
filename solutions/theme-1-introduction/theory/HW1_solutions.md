# Homework 1

## Solution 1.2

Change in notation:
$$
\text{Output of}\; Linear_1: \boldsymbol{z^{(1)}} \rightarrow \boldsymbol{s^{(1)}}.\\
\text{Output of}\; f: \boldsymbol{z^{(2)}} \rightarrow \boldsymbol{z^{(1)}}.\\
\text{Output of}\; Linear_2: \boldsymbol{z^{(3)}} \rightarrow \boldsymbol{s^{(2)}}.\\
\text{Output of}\; g: \text{Remains the same,}\; \boldsymbol{\hat{y}}.\\
$$

### Solution a)

1. `torch.nn.Linear`: $\text{Linear}(\boldsymbol{x})=W\boldsymbol{x}+b$.
2. `torch.nn.ReLU`: $\text{ReLU}(\boldsymbol{x}) = \max(\boldsymbol{0}, \boldsymbol{x})$.
3. `torch.nn.Linear`: $\text{Linear}(\boldsymbol{x})=W\boldsymbol{x}+b$.
4. `torch.nn.ReLU`: $\text{ReLU}(x) = \max(\boldsymbol{0}, \boldsymbol{x})$.
5. `torch.nn.MSELoss`: $l_{\text{MSE}}(\boldsymbol{\hat{y}}, \boldsymbol{y})=||\boldsymbol{\hat{y}}-\boldsymbol{y}||^2$.

### Solution b)

Strictly using $\boldsymbol{x}, \boldsymbol{y}, W^{(1)}, W^{(2)}, b^{(1)}, b^{(2)}$:

| Layer             | Input                                                        | Output                                                       |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| $\text{Linear}_1$ | $\boldsymbol{x}$                                             | $W^{(1)}\boldsymbol{x}+b^{(1)}$                              |
| $f$               | $W^{(1)}\boldsymbol{x}+b^{(1)}$                              | $\text{ReLU}(W^{(1)}\boldsymbol{x}+b^{(1)})$                 |
| $\text{Linear}_2$ | $\text{ReLU}(W^{(1)}\boldsymbol{x}+b^{(1)})$                 | $W^{(2)}\text{ReLU}(W^{(1)}\boldsymbol{x}+b^{(1)})+b^{(2)}$  |
| $g$               | $W^{(2)}\text{ReLU}(W^{(1)}\boldsymbol{x}+b^{(1)})+b^{(2)}$  | $\text{I}(W^{(2)}\text{ReLU}(W^{(1)}\boldsymbol{x}+b^{(1)})+b^{(2)})$ |
| $\text{Loss}$     | $\boldsymbol{\hat{y}}, \text{I}(W^{(2)}\text{ReLU}(W^{(1)}\boldsymbol{x}+b^{(1)})+b^{(2)})$ | $(\boldsymbol{\hat{y}} - \text{I}(W^{(2)}\text{ReLU}(W^{(1)}\boldsymbol{x}+b^{(1)})+b^{(2)}))(\boldsymbol{\hat{y}} - \text{I}(W^{(2)}\text{ReLU}(W^{(1)}\boldsymbol{x}+b^{(1)})+b^{(2)}))^{T}$ |

Using intermediate variables:

| Layer             | Input                                 | Output                                                       |
| ----------------- | ------------------------------------- | ------------------------------------------------------------ |
| $\text{Linear}_1$ | $\boldsymbol{x}$                      | $\boldsymbol{s^{(1)}}=W^{(1)}\boldsymbol{x}+b^{(1)}$         |
| f                 | $\boldsymbol{s^{(1)}}$                | $\boldsymbol{z^{(1)}}=\text{ReLU}(\boldsymbol{s^{(1)}})$     |
| $\text{Linear}_2$ | $\boldsymbol{z^{(1)}}$                | $\boldsymbol{s^{(2)}}=W^{(2)}\boldsymbol{z^{(1)}}+b^{(2)}$   |
| g                 | $\boldsymbol{s^{(2)}}$                | $\boldsymbol{\hat{y}}=\text{I}(\boldsymbol{s^{(2)}})$        |
| $\text{Loss}$     | $\boldsymbol{\hat{y}},\boldsymbol{y}$ | $\ell_{\text{MSE}}=(\boldsymbol{\hat{y}}-\boldsymbol{y})(\boldsymbol{\hat{y}}-\boldsymbol{y})^{T}$ |

Using components:

| Layer             | Input               | Output                                                      |
| ----------------- | ------------------- | ----------------------------------------------------------- |
| $\text{Linear}_1$ | $x_j$               | $s^{(1)}_i=\sum_{j}W^{(1)}_{ij}x_j+b^{(1)}_i$               |
| f                 | $s^{(1)}_i$         | $z^{(1)}_i=\text{ReLU}(s^{(1)}_i)$                          |
| $\text{Linear}_2$ | $z^{(1)}_i$         | $s^{(2)}_k=\sum_{i}W^{(2)}_{ki}z^{(1)}_{i}+b^{(2)}_{k}$     |
| g                 | $s^{(2)}_{k}$       | $y_{k}=g(s^{(2)}_{k})$                                      |
| $\text{Loss}$     | $\hat{y}_{k},y_{k}$ | $\ell_{\text{MSE}}=\sum_{k} (\hat{y}_k-y_k)(\hat{y}_k-y_k)$ |

### Solution c)

#### Dimensions

Following [numerator layout](https://en.wikipedia.org/wiki/Matrix_calculus):

$$
\boldsymbol{x} : d_{\boldsymbol{x}} \times 1. \\
\boldsymbol{s^{(1)}} : d_{\boldsymbol{s^{(1)}}} \times 1. \\
\boldsymbol{z^{(1)}} : d_{\boldsymbol{z^{(1)}}} \times 1. \\
\boldsymbol{s^{(2)}} : d_{\boldsymbol{s^{(2)}}} \times 1. \\
\boldsymbol{\hat{y}} : d_{\boldsymbol{\hat{y}}} \times 1. \\
W^{(1)} : d_{\boldsymbol{s^{(1)}}} \times d_{\boldsymbol{x}}. \\
W^{(2)} : d_{\boldsymbol{s^{(2)}}} \times d_\boldsymbol{z^{(1)}}. \\
b^{(1)} : d_{\boldsymbol{s^{(1)}}} \times 1. \\
b^{(2)} : d_{\boldsymbol{s^{(2)}}} \times 1. \\
\frac{\partial \ell}{\partial W^{(2)}} : d_{\boldsymbol{z^{(1)}}} \times d_\boldsymbol{s^{(2)}}. \\
\frac{\partial \ell}{\partial W^{(1)}} : d_{\boldsymbol{x}} \times d_\boldsymbol{s^{(1)}}. \\
\frac{\partial \ell}{\partial b^{(2)}} : 1 \times d_\boldsymbol{s^{(2)}}. \\
\frac{\partial \ell}{\partial b^{(1)}} : 1 \times d_\boldsymbol{s^{(1)}}.
$$

Where:
$$
d_{\boldsymbol{s^{(1)}}} = d_{\boldsymbol{z^{(1)}}}.\\
d_{\boldsymbol{s^{(2)}}} = d_{\boldsymbol{\hat{y}}}.
$$

#### Gradient of $W^{(2)}$

Using chain rule and tensor notation:
$$
\begin{align*}
\frac{\partial \ell}{\partial W^{(2)}_{ij}} &= 
\sum_{k, l}
\frac{\partial \ell}{\partial \hat{y}_{k}}
\frac{\partial \hat{y}_{k}}{\partial s_{l}^{(2)}}
\frac{\partial s_{l}^{(2)}}{\partial W^{(2)}_{ij}}. \\


&= 
\sum_{k,l}
\frac{\partial \ell}{\partial\hat{y}_{k}}
\frac{\partial\hat{y}_{k}}{\partial s_{l}^{(2)}}
\frac{\partial}{\partial W^{(2)}_{ij}}
\left(\sum_{m}W^{(2)}_{lm} z^{(1)}_{m} + b^{(2)}_{l}\right). \\

&= \sum_{k,l}
\frac{\partial \ell}{\partial \hat{y}_{k}}
\frac{\partial \hat{y}_{k}}{\partial s_{l}^{(2)}}z^{(1)}_{m}\delta_{il}\delta_{jm}. \\

&= 
\sum_{k}
\frac{\partial \ell}{\partial \hat{y}_{k}}
\frac{\partial\hat{y}_{k}}{\partial s_{i}^{(2)}}z^{(1)}_{j}. \\

&= 
\delta_{i}^{(2)}z^{(1)}_{j}.
\end{align*}
$$
In matrix form:
$$
\begin{align*}
\frac{\partial \ell}{\partial W^{(2)}} &= 
\begin{pmatrix}

\frac{\partial \ell}{\partial W^{(2)}_{00}} & 
\frac{\partial \ell}{\partial W^{(2)}_{10}} & 
\ldots & 
\frac{\partial L}{\partial W^{(2)}_{d_{\boldsymbol{s^{(2)}}}0}} \\

\frac{\partial \ell}{\partial W^{(2)}_{01}} & 
\frac{\partial \ell}{\partial W^{(2)}_{11}} & 
\ldots & 
\frac{\partial \ell}{\partial W^{(2)}_{d_{\boldsymbol{s^{(2)}}}1}} \\

\vdots & 
\vdots &
\ddots &
\vdots \\

\frac{\partial \ell}{\partial W^{(2)}_{0 d_{\boldsymbol{z^{(1)}}}}} & 
\frac{\partial \ell}{\partial W^{(2)}_{1 d_{\boldsymbol{z^{(1)}}}}} &
\ldots & 
\frac{\partial \ell}{\partial W^{(2)}_{ d_{\boldsymbol{s^{(2)}}}d_{\boldsymbol{z^{(1)}}}}}
\end{pmatrix}. \\

&= 
\begin{pmatrix}
	z^{(1)}_{0} \\
	
	\vdots \\
	
	z^{(1)}_{d_{\boldsymbol{z^{(1)}}}}
\end{pmatrix}

\begin{pmatrix} 
	\frac{\partial \ell}{\partial \hat{y}_{0}} & 
	\ldots & 
	\frac{\partial \ell}{\partial \hat{y}_{d_{\boldsymbol{\hat{y}}}}}
\end{pmatrix}
\begin{pmatrix} 
	\frac{\partial \hat{y}_{0}}{\partial s_{0}^{(2)}} & 
	\frac{\partial \hat{y}_{0}}{\partial s_{1}^{(2)}} & 
	\ldots & 
	\frac{\partial \hat{y}_{0}}{\partial s_{d_\boldsymbol{{s^{(2)}}}}^{(2)}} \\
	
	\frac{\partial \hat{y}_{1}}{\partial s_{0}^{(2)}} & 
	\frac{\partial \hat{y}_{1}}{\partial s_{1}^{(2)}} & 
	\ldots & 
	\frac{\partial \hat{y}_{1}}{\partial s_{d_\boldsymbol{{s^{(2)}}}}^{(2)}} \\
	
	\vdots & 
	\vdots &
	\ddots &
	\vdots \\
	
	\frac{\partial \hat{y}_{d_{\boldsymbol{\hat{y}}}}}{\partial s_{0}^{(2)}} & 
	\frac{\partial \hat{y}_{d_{\boldsymbol{\hat{y}}}}}{\partial s_{1}^{(2)}} &
	\ldots & 
	\frac{\partial \hat{y}_{d_{\boldsymbol{\hat{y}}}}}{\partial s_{d_{\boldsymbol{s^{(2)}}}}^{(2)}}
\end{pmatrix}
=\boldsymbol{z^{(1)}}
\frac{\partial \ell}{\partial\boldsymbol{\hat{y}}} 
\frac{\partial \boldsymbol{\hat{y}}}{\partial \boldsymbol{s^{(2)}}}. \\

&=
\begin{pmatrix}
	z^{(1)}_{0} \\
	
	\vdots \\
	
	z^{(1)}_{d_{\boldsymbol{z^{(1)}}}}
\end{pmatrix}

\begin{pmatrix} 
	\delta^{(2)}_{0} & 
	\ldots & 
	\delta^{(2)}_{d_\boldsymbol{s^{(2)}}}
\end{pmatrix}
=\boldsymbol{z^{(1)}}[\boldsymbol{\delta}^{(2)}]^T. \\
\end{align*}
$$


This results are for any activation function and any loss, in our case:
$$
\begin{align*}
\frac{\partial\hat{y}_{k}}{\partial s_{i}^{(2)}} &= 
\frac{\partial}{\partial s_{i}^{(2)}}\text{g}\left(s_{k}^{(2)}\right) =
\delta_{ki}. \\ \\

\frac{\partial\boldsymbol{\hat{y}}}{\partial\boldsymbol{s^{(2)}}} &= 
I_{d_{\boldsymbol{\hat{y}}} \times d_{\boldsymbol{s^{(2)}}}}.
\end{align*}
$$
And for the loss:
$$
\begin{align*}
\frac{\partial \ell}{\partial \hat{y}_{k}} &= 
\frac{\partial}{\partial \hat{y}_{k}} \left[\sum_{i}(\hat{y}_{i}-y_i)^{2} \right]. \\

&=
\sum_i 2(\hat{y}_{i}-y_i)\delta_{ik}. \\

&=
2(\hat{y}_{k}-y_k). \\  \\

\frac{\partial L}{\partial \boldsymbol{\hat{y}}} &= 
2(\boldsymbol{\hat{y}}-\boldsymbol{y})^{\text{T}}.
\end{align*}
$$

Inserting that in the formula for $\boldsymbol{\delta^{(2)}}$:
$$
\delta_{i}^{(2)} = 2(\hat{y}_{i}-y_i). \\

\boldsymbol{\delta^{(2)}} = 
2 (\boldsymbol{\hat{y}} - \boldsymbol{y}) =
2
\begin{pmatrix}
\hat{y}_{0}-y_0 \\
\vdots \\
\hat{y}_{d_\boldsymbol{s^{(2)}}}-y_{d_\boldsymbol{s^{(2)}}}
\end{pmatrix}.
$$
And for $\frac{\partial \ell}{\partial W^{(2)}}$:
$$
\begin{align*}
\frac{\partial \ell}{\partial W^{(2)}_{ij}} &= 
2(\hat{y}_{i}-y_i)z^{(1)}_{j}. \\

\frac{\partial \ell}{\partial W^{(2)}} &=
2
\begin{pmatrix}
	z^{(1)}_{0} \\
	
	\vdots \\
	
	z^{(1)}_{d_{\boldsymbol{z^{(1)}}}}
\end{pmatrix}
\begin{pmatrix} 
	\hat{y}_{0}-y_0 & 
	\ldots & 
	\hat{y}_{d_\boldsymbol{s^{(2)}}}-y_{d_\boldsymbol{s^{(2)}}}
\end{pmatrix}. \\

&=
2\boldsymbol{z^{(1)}}
(\boldsymbol{\hat{y}}-\boldsymbol{y})^{T}
\end{align*}
$$

#### Gradient of $b^{(2)}$

Using chain rule and components and having into account the previous results for $W^{(2)}$:
$$
\begin{align*}
\frac{\partial \ell}{\partial b^{(2)}_{i}} &= 
\sum_{k, l}
\frac{\partial \ell}{\partial \hat{y}_{k}}
\frac{\partial \hat{y}_{k}}{\partial s_{l}^{(2)}}
\frac{\partial s_{l}^{(2)}}{\partial b^{(2)}_{i}}. \\

&= 
\sum_{k,l}
\frac{\partial l}{\partial \hat{y}_{k}}
\frac{\partial \hat{y}_{k}}{\partial s_{l}^{(2)}}
\frac{\partial}{\partial b^{(2)}_{i}} \left(\sum_{m}W^{(2)}_{lm}z^{(1)}_{m} + b^{(2)}_{l}\right). \\

&= 
\sum_{k,l}
\frac{\partial \ell}{\partial \hat{y}_{k}}
\frac{\partial \hat{y}_{k}}{\partial s_{l}^{(2)}}\delta_{il}. \\

&= 
\sum_{k}
\frac{\partial \ell}{\partial \hat{y}_{k}}
\frac{\partial \hat{y}_{k}}{\partial s_{i}^{(2)}}. \\

&=
\delta_{i}^{(2)}. \\

&=
2(\hat{y}_{i}-y_i).
\end{align*}
$$

In matrix form:
$$
\begin{align*}
\frac{\partial L}{\partial b^{(2)}} &= 
\begin{pmatrix}

\frac{\partial \ell}{\partial b^{(2)}_{0}} & 
\frac{\partial \ell}{\partial b^{(2)}_{1}} & 
\ldots \\
\end{pmatrix} \\

&= 
\begin{pmatrix} 
	\frac{\partial \ell}{\partial \hat{y}_{0}} & 
	\ldots & 
	\frac{\partial \ell}{\partial \hat{y}_{d_{\boldsymbol{\hat{y}}}}}
\end{pmatrix}
\begin{pmatrix} 
	\frac{\partial \hat{y}_{0}}{\partial s_{0}^{(2)}} & 
	\frac{\partial \hat{y}_{0}}{\partial s_{1}^{(2)}} & 
	\ldots & 
	\frac{\partial \hat{y}_{0}}{\partial s_{d_\boldsymbol{{s^{(2)}}}}^{(2)}} \\
	
	\frac{\partial \hat{y}_{1}}{\partial s_{0}^{(2)}} & 
	\frac{\partial \hat{y}_{1}}{\partial s_{1}^{(2)}} & 
	\ldots & 
	\frac{\partial \hat{y}_{1}}{\partial s_{d_\boldsymbol{{s^{(2)}}}}^{(2)}} \\
	
	\vdots & 
	\vdots &
	\ddots &
	\vdots \\
	
	\frac{\partial \hat{y}_{d_{\boldsymbol{\hat{y}}}}}{\partial s_{0}^{(2)}} & 
	\frac{\partial \hat{y}_{d_{\boldsymbol{\hat{y}}}}}{\partial s_{1}^{(2)}} &
	\ldots & 
	\frac{\partial \hat{y}_{d_{\boldsymbol{\hat{y}}}}}{\partial s_{d_{\boldsymbol{s^{(2)}}}}^{(2)}}
\end{pmatrix}. \\

&=
\frac{\partial \ell}{\partial\boldsymbol{\hat{y}}} 
\frac{\partial \boldsymbol{\hat{y}}}{\partial \boldsymbol{s^{(2)}}}. \\

&=
[\boldsymbol{\delta}^{(2)}]^T.\\

&=
2(\boldsymbol{\hat{y}}-\boldsymbol{y})^{T}. \\

&=
2
\begin{pmatrix} 
	\hat{y}_{0}-y_0 & 
	\ldots & 
	\hat{y}_{d_\boldsymbol{s^{(2)}}}-y_{d_\boldsymbol{s^{(2)}}}
\end{pmatrix}. \\
\end{align*}
$$

#### Gradient of $W^{(1)}$

Using chain rule and tensor notation:
$$
\begin{align*}
\frac{\partial \ell}{\partial W^{(1)}_{ij}} &= 
\sum_{k, l, m, n}
\frac{\partial \ell}{\partial \hat{y}_{k}}
\frac{\partial \hat{y}_{k}}{\partial s_{l}^{(2)}}
\frac{\partial s_{l}^{(2)}}{\partial z_{m}^{(1)}}
\frac{\partial z_{m}^{(1)}}{\partial s_{n}^{(1)}}
\frac{\partial s_{n}^{(1)}}{\partial W^{(1)}_{ij}}. \\

&=
\sum_{l, m, n}
\delta^{(2)}_{l}
\frac{\partial s_{l}^{(2)}}{\partial z_{m}^{(1)}}
\frac{\partial z_{m}^{(1)}}{\partial s_{n}^{(1)}}
\frac{\partial s_{n}^{(1)}}{\partial W^{(1)}_{ij}}. \\

&=
\sum_{n}
\delta_n^{(1)}\frac{\partial s_{n}^{(1)}}{\partial W^{(1)}_{ij}}. \\

&=
\sum_{n}
\delta_i^{(1)}x_j. \\ \\

\frac{\partial \ell}{\partial W^{(1)}} &=
\boldsymbol{x}\left[\boldsymbol{\delta^{(1)}}\right]^{T}.
\end{align*}
$$
Where $\boldsymbol{\delta^{(L=1)}}$ are the so called "errors" for the linear layer $L=1$. Then, we can compute $\frac{\partial \ell}{\partial W^{(1)}}$ in terms of the jacobians:
$$
\begin{align*}
\frac{\partial \ell}{\partial W^{(1)}} &=
\boldsymbol{x}\left[\boldsymbol{\delta^{(1)}}\right]^{T}. \\

\boldsymbol{\delta^{(1)}}=
\left[
\frac{\partial \boldsymbol{s^{(2)}}}{\partial \boldsymbol{z^{(1)}}}
\frac{\partial \boldsymbol{z^{(1)}}}{\partial \boldsymbol{s^{(1)}}}
\right]^{T}
&\rightarrow 
\boldsymbol{x}
\left[
\left[
\frac{\partial \boldsymbol{s^{(2)}}}{\partial \boldsymbol{z^{(1)}}}
\frac{\partial \boldsymbol{z^{(1)}}}{\partial \boldsymbol{s^{(1)}}}
\right]^{T}
\boldsymbol{\delta^{(2)}}
\right]^{T}. \\


\boldsymbol{\delta^{(2)}} = 
\left[
\frac{\partial \ell}{\partial\boldsymbol{\hat{y}}} 
\frac{\partial \boldsymbol{\hat{y}}}{\partial \boldsymbol{s^{(2)}}}
\right]^{T}
&\rightarrow 
\boldsymbol{x}
\frac{\partial \ell}{\partial\boldsymbol{\hat{y}}} 
\frac{\partial \boldsymbol{\hat{y}}}{\partial \boldsymbol{s^{(2)}}}
\frac{\partial \boldsymbol{s^{(2)}}}{\partial \boldsymbol{z^{(1)}}}
\frac{\partial \boldsymbol{z^{(1)}}}{\partial \boldsymbol{s^{(1)}}}.
\end{align*}
$$

#### Gradient of $W^{(L)}$

The errors are easily generalizable:
$$
\delta^{(L)}_i =
\sum_{p,q}
\delta^{(L+1)}_{p}
\frac{\partial s_{p}^{(L+1)}}{\partial z_{q}^{(L)}}
\frac{\partial z_{q}^{(L)}}{\partial s_{i}^{(L)}}.
$$
In matrix form:
$$
\begin{align*}
\boldsymbol{\delta}^{(L)} &=
\left[
\begin{pmatrix}
\delta^{(L+1)}_0 &
\ldots &
\delta^{(L+1)}_{d_{\boldsymbol{s^{(2)}}}}
\end{pmatrix}

\begin{pmatrix}
\frac{\partial s_{0}^{(L+1)}}{\partial z_{0}^{(L)}} & 
\frac{\partial s_{0}^{(L+1)}}{\partial z_{1}^{(L)}} & 
\ldots & 
\frac{\partial s_{0}^{(L+1)}}{\partial z_{d_{\boldsymbol{z^{(L)}}}}^{(L)}} \\

\frac{\partial s_{1}^{(L+1)}}{\partial z_{0}^{(L)}} & 
\frac{\partial s_{1}^{(L+1)}}{\partial z_{1}^{(L)}} & 
\ldots & 
\frac{\partial s_{1}^{(L+1)}}{\partial z_{d_{\boldsymbol{z^{(L)}}}}^{(L)}} \\

\vdots & 
\vdots &
\ddots &
\vdots \\

\frac{\partial s_{d_{\boldsymbol{s^{(L+1)}}}}^{(L+1)}}{\partial z_{0}^{(L)}} & 
\frac{\partial s_{d_{\boldsymbol{s^{(L+1)}}}}^{(L+1)}}{\partial z_{1}^{(L)}} &
\ldots & 
\frac{\partial s_{d_{\boldsymbol{s^{(L+1)}}}}^{(L+1)}}{\partial z_{d_{\boldsymbol{z^{(L)}}}}^{(L)}}
\end{pmatrix}

\begin{pmatrix}
\frac{\partial z_{0}^{(L)}}{\partial s_{0}^{(L)}} & 
\frac{\partial z_{0}^{(L)}}{\partial s_{1}^{(L)}} & 
\ldots & 
\frac{\partial z_{0}^{(L)}}{\partial s_{d_{\boldsymbol{s^{(L)}}}}^{(L)}} \\

\frac{\partial z_{1}^{(L)}}{\partial s_{0}^{(L)}} & 
\frac{\partial z_{1}^{(L)}}{\partial s_{1}^{(L)}} & 
\ldots & 
\frac{\partial z_{1}^{(L)}}{\partial s_{d_{\boldsymbol{s^{(L)}}}}^{(L)}} \\

\vdots & 
\vdots &
\ddots &
\vdots \\

\frac{\partial z_{d_{\boldsymbol{z^{(L)}}}}^{(L)}}{\partial s_{0}^{(L)}} & 
\frac{\partial z_{d_{\boldsymbol{z^{(L)}}}}^{(L)}}{\partial s_{1}^{(L)}} &
\ldots & 
\frac{\partial z_{d_{\boldsymbol{z^{(L)}}}}^{(L)}}{\partial s_{d_{\boldsymbol{s^{(L)}}}}^{(L)}}
\end{pmatrix}
\right]^{T}. \\

&=
\left[
\left[\boldsymbol{\delta^{(L+1)}}\right]^{T}
\frac{\partial \boldsymbol{s^{(L+1)}}}{\partial \boldsymbol{z^{(L)}}}
\frac{\partial \boldsymbol{z^{(L)}}}{\partial \boldsymbol{s^{(L)}}}
\right]^{T}.\\

&=
\left[
\frac{\partial \boldsymbol{s^{(L+1)}}}{\partial \boldsymbol{z^{(L)}}}
\frac{\partial \boldsymbol{z^{(L)}}}{\partial \boldsymbol{s^{(L)}}}
\right]^{T}
\boldsymbol{\delta^{(L+1)}}.
\end{align*}
$$

Now, let's compute $\frac{\partial \boldsymbol{s^{(L+1)}}}{\partial \boldsymbol{z^{(L)}}}$ for a linear layer:
$$
\begin{align*}
\frac{\partial s_{i}^{(L+1)}}{\partial z_{j}^{(L)}} &=
\frac{\partial}{\partial z_{j}^{(L)}} \left(\sum_{k} W^{(L+1)}_{ik} z^{(L)}_{k} + b^{(L+1)}_{i}\right). \\

\frac{\partial s_{i}^{(L+1)}}{\partial z_{j}^{(L)}} &= 
W^{(L+1)}_{ij}. \\ \\

\frac{\partial \boldsymbol{s^{(L+1)}}}{\partial \boldsymbol{z^{(L)}}} &= 
W^{(L+1)}.
\end{align*}
$$


Taking into account the previous expressions, we can compute the gradient for any linear layer and any activation function:
$$
\frac{\partial \ell}{\partial W^{(L)}} =
\boldsymbol{z^{(L-1)}}\left[\boldsymbol{\delta^{(L)}}\right]^{T}.\\


\boldsymbol{\delta^{(L)}} = 
\left[
W^{(L+1)}
\frac{\partial \boldsymbol{z^{(L)}}}{\partial \boldsymbol{s^{(L)}}}
\right]^{T}
\boldsymbol{\delta^{(L+1)}}.\\

\boldsymbol{z^{(0)}} =
\boldsymbol{x}. \\

\boldsymbol{\delta^{(L_{\max})}} =
\left[
\frac{\partial \ell}{\partial\boldsymbol{\hat{y}}} 
\frac{\partial \boldsymbol{\hat{y}}}{\partial \boldsymbol{s^{(L_{\max})}}}
\right]^{T}.
$$

In regard to $\frac{\partial \boldsymbol{z^{(L)}}}{\partial \boldsymbol{s^{(L)}}}$ , we can compute it for $g=I(\cdot)$ and $f=\text{ReLU}(\cdot)$ (one of the most common cases): 
$$
f=
\text{ReLU}(\cdot) \rightarrow \frac{\partial z_{i}^{(L)}}{\partial s_{j}^{(L)}} = 
\max(0, \text{sign}(s^{L}_j))\delta_{ij}.\\

\frac{\partial \boldsymbol{z^{(L)}}}{\partial \boldsymbol{s^{(L)}}} = 
I^{+\boldsymbol{s^{(L)}}}_{\boldsymbol{z^{(L)}} \times \boldsymbol{s^{(L)}}} =

\begin{pmatrix}
\max(0, \text{sign}(s^{(L)}_0)) &
0 &
\ldots &
0 \\

0 &
\max(0, \text{sign}(s^{(L)}_1)) &
\ldots &
0 \\

\vdots &
\vdots &
\ddots &
\vdots \\

0 &
0 &
\ldots &
\max(0, \text{sign}(s^{(L)}_{d_{\boldsymbol{s^{(L)}}}})) 
\end{pmatrix} \\ \\

g=
I(\cdot) \rightarrow \frac{\partial z_{i}^{(L)}}{\partial s_{j}^{(L)}} = 
\delta_{ij}.\\
\frac{\partial \boldsymbol{z^{(L)}}}{\partial \boldsymbol{s^{(L)}}} = 
I_{\boldsymbol{z^{(L)}} \times \boldsymbol{s^{(L)}}} =
\begin{pmatrix}
1 &
0 &
\ldots &
0 \\

0 &
1 &
\ldots &
0 \\

\vdots &
\vdots &
\ddots &
0 \\

0 &
0 &
\ldots &
1
\end{pmatrix}.
$$

Then, the "errors" for any linear layer are given by:
$$
\begin{align*}
\delta^{(L)}_i &=
\sum_{p,q}
\delta^{(L+1)}_{p}
\frac{\partial z_{q}^{(L)}}{\partial s_{i}^{(L)}}
\frac{\partial}{\partial z_{q}^{(L)}} \left(\sum_{l} W^{(L+1)}_{pl} z^{(L)}_{l} + b^{(L+1)}_{p}\right). \\

&=
\sum_{p,q}
\delta^{(L+1)}_{p}
W^{(L+1)}_{pq}
\frac{\partial z_{q}^{(L)}}{\partial s_{i}^{(L)}}
. \\ \\

\boldsymbol{\delta^{(L)}} &=
\left[
W^{L+1}
\frac{\partial \boldsymbol{z^{(L)}}}{\partial \boldsymbol{s^{(L)}}}
\right]^{T}
\boldsymbol{\delta^{(L+1)}}.


\end{align*}
$$
For a $\text{ReLU}(\cdot)$:
$$
\boldsymbol{\delta^{(L)}} =
\left[
W^{L+1}
I^{+\boldsymbol{s^{(L)}}}_{\boldsymbol{z^{(L)}} \times \boldsymbol{s^{(L)}}}
\right]^{T}
\boldsymbol{\delta^{(L+1)}}.
$$
As an example, let's particularize our computation of $\frac{\partial \ell}{\partial W^{(1)}}$:
$$
\begin{align*}
\frac{\partial L}{\partial W^{(1)}} &=
\boldsymbol{x}\left[\boldsymbol{\delta^{(1)}}\right]^{T}. \\

&=
\boldsymbol{x}
\left[
\left[
W^{(2)}
I^{+\boldsymbol{s^{(1)}}}_{\boldsymbol{z^{(1)}} \times \boldsymbol{s^{(1)}}}
\right]^{T}
\boldsymbol{\delta^{(2)}}
\right]^{T}. \\

&=
2
\boldsymbol{x}
(\boldsymbol{\hat{y}} - \boldsymbol{y})^{T}
W^{(2)}
I^{+\boldsymbol{s^{(1)}}}_{\boldsymbol{z^{(1)}} \times \boldsymbol{s^{(1)}}}.
\end{align*}
$$
#### Gradient of $b^{(1)}$

Following the same idea:
$$
\begin{align*}
\frac{\partial \ell}{\partial b^{(1)}_{i}} &= 
\sum_{k, l, m, n}
\frac{\partial \ell}{\partial \hat{y}_{k}}
\frac{\partial \hat{y}_{k}}{\partial s_{l}^{(2)}}
\frac{\partial s_{l}^{(2)}}{\partial z_{m}^{(1)}}
\frac{\partial z_{m}^{(1)}}{\partial s_{n}^{(1)}}
\frac{\partial s_{n}^{(1)}}{\partial b^{(1)}_{i}}. \\

&=
\sum_{n}
\delta_n^{(1)}\frac{\partial s_{n}^{(1)}}{\partial b^{(1)}_{i}}. \\

&=
\delta_i^{(1)}. \\ \\

\frac{\partial \ell}{\partial b^{(1)}} &=
\left[\boldsymbol{\delta^{(1)}}\right]^{T}.
\end{align*}
$$
In terms of the jacobians:
$$
\frac{\partial \ell}{\partial b^{(1)}} =
\frac{\partial \ell}{\partial\boldsymbol{\hat{y}}} 
\frac{\partial \boldsymbol{\hat{y}}}{\partial \boldsymbol{s^{(2)}}}
\frac{\partial \boldsymbol{s^{(2)}}}{\partial \boldsymbol{z^{(1)}}}
\frac{\partial \boldsymbol{z^{(1)}}}{\partial \boldsymbol{s^{(1)}}}.
$$

#### Gradient of $b^{(L)}$

For any linear layer, the gradient respect to the bias is:
$$
\begin{align*}
\frac{\partial \ell}{\partial b^{(L)}_{i}} &= 
\delta_i^{(L)}. \\ \\

\frac{\partial \ell}{\partial b^{(L)}} &=
\left[\boldsymbol{\delta^{(L)}}\right]^{T}.
\end{align*}
$$
Where $\boldsymbol{\delta^{(L_{\max})}}$ is given in the previous section. Let's particularize for our special case:
$$
\frac{\partial L}{\partial b^{(1)}} =
2(\boldsymbol{\hat{y}} - \boldsymbol{y})^{T}
W^{(2)}
I^{+\boldsymbol{s^{(1)}}}_{\boldsymbol{z^{(1)}} \times \boldsymbol{s^{(1)}}}.
$$

#### Summary

Shapes:
$$
\boldsymbol{s^{(L)}} = 1 \times d_{\boldsymbol{s^{(L)}}}.\\
\boldsymbol{z^{(L)}} = 1 \times d_{\boldsymbol{z^{(L)}}}.\\
W^{(L)} : d_{\boldsymbol{s^{(L)}}} \times d_\boldsymbol{z^{(L-1)}}. \\
b^{(L)} : d_{\boldsymbol{s^{(L)}}} \times 1. \\
\frac{\partial \ell}{\partial W^{(L)}} : d_{\boldsymbol{z^{(L-1)}}} \times d_\boldsymbol{s^{(L)}}. \\
\frac{\partial \ell}{\partial b^{(L)}} : 1 \times d_\boldsymbol{s^{(L)}}. \\
$$
Where:
$$
d_{\boldsymbol{z^{(0)}}} = d_{\boldsymbol{x}}.\\
d_{\boldsymbol{z^{(L_{\max})}}} = d_{\boldsymbol{\hat{y}}} = d_{\boldsymbol{y}}.
$$
Backpropagation for a stack of linear layers in matrix form:
$$
\frac{\partial \ell}{\partial W^{(L)}} =
\boldsymbol{z^{(L-1)}}\left[\boldsymbol{\delta^{(L)}}\right]^{T}.\\


\boldsymbol{\delta^{(L)}} = 
\left[
W^{(L+1)}
\frac{\partial \boldsymbol{z^{(L)}}}{\partial \boldsymbol{s^{(L)}}}
\right]^{T}
\boldsymbol{\delta^{(L+1)}}. \\

\frac{\partial \ell}{\partial b^{(L)}} =
\left[\boldsymbol{\delta^{(L)}}\right]^{T}. \\

\boldsymbol{z^{(0)}} =
\boldsymbol{x}. \\

\boldsymbol{\delta^{(L_{\max})}} =
\left[
\frac{\partial \ell}{\partial\boldsymbol{\hat{y}}} 
\frac{\partial \boldsymbol{\hat{y}}}{\partial \boldsymbol{s^{(L_{\max})}}}
\right]^{T}.
$$
$\frac{\partial \boldsymbol{z^{(L)}}}{\partial \boldsymbol{s^{(L)}}}$ for a $\text{ReLU}(\cdot)$:
$$
\frac{\partial \boldsymbol{z^{(L)}}}{\partial \boldsymbol{s^{(L)}}} = 
I^{+\boldsymbol{s^{(L)}}}_{\boldsymbol{z^{(L)}} \times \boldsymbol{s^{(L)}}} =

\begin{pmatrix}
\max(0, \text{sign}(s^{L}_0)) &
0 &
\ldots &
0 \\

0 &
\max(0, \text{sign}(s^{L}_1)) &
\ldots &
0 \\

\vdots &
\vdots &
\ddots &
0 \\

0 &
0 &
\ldots &
\max(0, \text{sign}(s^{L}_{d_{\boldsymbol{s^{(L)}}}}))
\end{pmatrix}.
$$


| Parameter | Gradient                                                     | Gradient shape                                            |
| :-------- | :----------------------------------------------------------- | :-------------------------------------------------------- |
| $W^{(1)}$ | $\boldsymbol{x}\frac{\partial \ell}{\partial\boldsymbol{\hat{y}}} \frac{\partial \boldsymbol{\hat{y}}}{\partial \boldsymbol{s^{(2)}}}\frac{\partial \boldsymbol{s^{(2)}}}{\partial \boldsymbol{z^{(1)}}}\frac{\partial \boldsymbol{z^{(1)}}}{\partial \boldsymbol{s^{(1)}}}=2\boldsymbol{x}(\boldsymbol{\hat{y}} - \boldsymbol{y})^{T} W^{(2)}I^{+\boldsymbol{s^{(1)}}}_{\boldsymbol{z^{(1)}} \times \boldsymbol{s^{(1)}}}.$ | $d_{\boldsymbol{x}} \times d_{\boldsymbol{s^{(1)}}}.$     |
| $b^{(1)}$ | $\frac{\partial \ell}{\partial\boldsymbol{\hat{y}}} \frac{\partial \boldsymbol{\hat{y}}}{\partial \boldsymbol{s^{(2)}}}\frac{\partial \boldsymbol{s^{(2)}}}{\partial \boldsymbol{z^{(1)}}}\frac{\partial \boldsymbol{z^{(1)}}}{\partial \boldsymbol{s^{(1)}}}=2(\boldsymbol{\hat{y}} - \boldsymbol{y})^{T}W^{(2)} I^{+\boldsymbol{s^{(1)}}}_{\boldsymbol{z^{(1)}} \times \boldsymbol{s^{(1)}}}.$ | $1 \times d_{\boldsymbol{s^{(1)}}}.$                      |
| $W^{(2)}$ | $\boldsymbol{z^{(1)}}\frac{\partial \ell}{\partial\boldsymbol{\hat{y}}}\frac{\partial \boldsymbol{\hat{y}}}{\partial \boldsymbol{s^{(2)}}}=2\boldsymbol{z^{(1)}}(\boldsymbol{\hat{y}}-\boldsymbol{y})^{T}.$ | $d_\boldsymbol{z^{(1)}} \times d_{\boldsymbol{s^{(2)}}}.$ |
| $b^{(2)}$ | $\frac{\partial \ell}{\partial\boldsymbol{\hat{y}}}\frac{\partial \boldsymbol{\hat{y}}}{\partial \boldsymbol{s^{(2)}}}=2(\boldsymbol{\hat{y}}-\boldsymbol{y})^{T}.$ | $1\times d_{\boldsymbol{s^{(2)}}}.$                       |

### Solution d)

With the change in notation:
$$
\frac{\partial \boldsymbol{z^{(2)}}}{\partial \boldsymbol{z^{(1)}}} \rightarrow 
\frac{\partial \boldsymbol{z^{(1)}}}{\partial \boldsymbol{s^{(1)}}}.\\

\frac{\partial \boldsymbol{\hat{y}}}{\partial \boldsymbol{z^{(3)}}} \rightarrow 
\frac{\partial \boldsymbol{\hat{y}}}{\partial \boldsymbol{s^{(2)}}}.\\
$$
$\frac{\partial \boldsymbol{z^{(1)}}}{\partial \boldsymbol{s^{(1)}}}$:
$$
f=
\text{ReLU}(\cdot) \rightarrow 
\frac{\partial z^{(1)}_i}{\partial s_{j}^{(1)}} = 
\frac{\partial}{\partial s_{j}^{(1)}} \text{ReLU}(s^{(1)}_i) =
\max(0, \text{sign}(s^{(1)}_j)) \delta_{ij}.\\

\frac{\partial \boldsymbol{z^{(1)}}}{\partial \boldsymbol{s^{(1)}}} = 
I^{+\boldsymbol{s^{(1)}}}_{d_{\boldsymbol{z^{(1)}}} \times \boldsymbol{s^{(1)}}} =

\begin{pmatrix}
\max(0, \text{sign}(s^{(1)}_0)) &
0 &
\ldots &
0 \\

0 &
\max(0, \text{sign}(s^{(1)}_1)) &
\ldots &
0 \\

\vdots &
\vdots &
\ddots &
\vdots \\

0 &
0 &
\ldots &
\max(0, \text{sign}(s^{(1)}_{d_{\boldsymbol{s^{(1)}}}}))
\end{pmatrix}.
$$
$\frac{\partial \boldsymbol{\hat{y}}}{\partial \boldsymbol{s^{(2)}}}$:
$$
g =
I(\cdot) \rightarrow \frac{\partial \hat{y}_i}{\partial s_{j}^{(2)}} = 
\frac{\partial}{\partial s_{j}^{(2)}} I(s^{(2)}_i)=
\delta_{ij},
\;\; i=0,...,d_{\boldsymbol{\hat{y}}},\; j=0,...,d_{\boldsymbol{s^{(2)}}}.\\


\frac{\partial \boldsymbol{\boldsymbol{\hat{y}}}}{\partial \boldsymbol{s^{(2)}}} = 
I_{d_\boldsymbol{\hat{y}} \times \boldsymbol{s^{(2)}}} =
\begin{pmatrix}
1 &
0 &
\ldots &
0 \\

0 &
1 &
\ldots &
0 \\

\vdots &
\vdots &
\ddots &
0 \\

0 &
0 &
\ldots &
1
\end{pmatrix}. \\ \\
$$
$\frac{\partial \ell}{\partial \boldsymbol{\hat{y}}}$:
$$
\begin{align*}
\frac{\partial \ell}{\partial \hat{y}_{i}} &= 
\frac{\partial}{\partial \hat{y}_{i}} \left[\sum_{j}(\hat{y}_{j}-y_j)^{2} \right]. \\

&=
\sum_j 2(\hat{y}_{j}-y_j)\delta_{ij}. \\

&=
2(\hat{y}_{i}-y_i). \\  \\

\frac{\partial \ell}{\partial \boldsymbol{\hat{y}}} &= 
2(\boldsymbol{\hat{y}}-\boldsymbol{y})^{\text{T}}. \\

&=
2
\begin{pmatrix}
\hat{y}_{0}-y_0 &
\ldots &
\hat{y}_{d_{\boldsymbol{\hat{y}}}}-y_{d_{\boldsymbol{y}}}
\end{pmatrix}
\end{align*}
$$

## Solution 1.3

#### Solution a)

In the case of **b)** the loss function (the replacement is done in the table with intermediate variables only) :

| Layer             | Input                                 | Output                                                       |
| ----------------- | ------------------------------------- | ------------------------------------------------------------ |
| $\text{Linear}_1$ | $\boldsymbol{x}$                      | $\boldsymbol{s^{(1)}}=W^{(1)}\boldsymbol{x}+b^{(1)}$         |
| $\sigma$          | $\boldsymbol{s^{(1)}}$                | $\boldsymbol{z^{(1)}}=\sigma(\boldsymbol{s^{(1)}})$          |
| $\text{Linear}_2$ | $\boldsymbol{z^{(1)}}$                | $\boldsymbol{s^{(2)}}=W^{(2)}\boldsymbol{z^{(1)}}+b^{(2)}$   |
| $\sigma$          | $\boldsymbol{s^{(2)}}$                | $\boldsymbol{\hat{y}}=\sigma(\boldsymbol{s^{(2)}})$          |
| $\text{Loss}$     | $\boldsymbol{\hat{y}},\boldsymbol{y}$ | $\ell_{\text{MSE}}=(\boldsymbol{\hat{y}}-\boldsymbol{y})(\boldsymbol{\hat{y}}-\boldsymbol{y})^{T}$ |

In the case of **c)** the jacobians $\frac{\partial \boldsymbol{z^{(1)}}}{\partial \boldsymbol{s^{(1)}}}$ and $\frac{\partial \boldsymbol{\hat{y}}}{\partial \boldsymbol{s^{(2)}}}$.

In the case of **d)** , we need to compute the derivatives so we can see the components explicitly, the derivative of $\sigma$ is:
$$
\sigma^{\prime} = \sigma(1-\sigma).
$$
Then, $\frac{\partial \boldsymbol{z^{(1)}}}{\partial \boldsymbol{s^{(1)}}}$:
$$
f=
\sigma(\cdot) \rightarrow 
\frac{\partial z^{(1)}_i}{\partial s_{j}^{(1)}} = 
\frac{\partial}{\partial s_{j}^{(1)}} \sigma(s^{(1)}_i) =
\sigma(s^{(1)}_i)(1-\sigma(s^{(1)}_i)) \delta_{ij}.\\

\frac{\partial \boldsymbol{z^{(1)}}}{\partial \boldsymbol{s^{(1)}}} = 

\begin{pmatrix}
\sigma(s^{(1)}_0)(1-\sigma(s^{(1)}_0)) &
0 &
\ldots &
0 \\

0 &
\sigma(s^{(1)}_1)(1-\sigma(s^{(1)}_1)) &
\ldots &
0 \\

\vdots &
\vdots &
\ddots &
\vdots \\

0 &
0 &
\ldots &
\sigma(s^{(1)}_{d_{\boldsymbol{s^{(1)}}}})(1-\sigma(s^{(1)}_{d_{\boldsymbol{s^{(1)}}}}))
\end{pmatrix}.
$$
$\frac{\partial \boldsymbol{\hat{y}}}{\partial \boldsymbol{s^{(2)}}}$:
$$
g=
\sigma(\cdot) \rightarrow 
\frac{\partial \hat{y}_i}{\partial s_{j}^{(2)}} = 
\frac{\partial}{\partial s_{j}^{(2)}} \sigma(s^{(2)}_i) =
\sigma(s^{(2)}_i)(1-\sigma(s^{(2)}_i)) \delta_{ij}.\\

\frac{\partial \boldsymbol{\hat{y}}}{\partial \boldsymbol{s^{(2)}}} = 


\begin{pmatrix}
\sigma(s^{(2)}_0)(1-\sigma(s^{(2)}_0)) &
0 &
\ldots &
0 \\

0 &
\sigma(s^{(2)}_1)(1-\sigma(s^{(2)}_1)) &
\ldots &
0 \\

\vdots &
\vdots &
\ddots &
\vdots \\

0 &
0 &
\ldots &
\sigma(s^{(2)}_{d_{\boldsymbol{s^{(2)}}}})(1-\sigma(s^{(2)}_{d_{\boldsymbol{s^{(2)}}}}))
\end{pmatrix}.
$$
$\frac{\partial \ell}{\partial \boldsymbol{\hat{y}}}$ remains the same.

#### Solution **b)**

In the equations of **b)** only the loss function, $\ell_{\text{MSE}} \rightarrow \ell_{\text{BCE}}$

| Layer             | Input                                 | Output                                                       |
| ----------------- | ------------------------------------- | ------------------------------------------------------------ |
| $\text{Linear}_1$ | $\boldsymbol{x}$                      | $\boldsymbol{s^{(1)}}=W^{(1)}\boldsymbol{x}+b^{(1)}$         |
| $\sigma$          | $\boldsymbol{s^{(1)}}$                | $\boldsymbol{z^{(1)}}=\sigma(\boldsymbol{s^{(1)}})$          |
| $\text{Linear}_2$ | $\boldsymbol{z^{(1)}}$                | $\boldsymbol{s^{(2)}}=W^{(2)}\boldsymbol{z^{(1)}}+b^{(2)}$   |
| $\sigma$          | $\boldsymbol{s^{(2)}}$                | $\boldsymbol{\hat{y}}=\sigma(\boldsymbol{s^{(2)}})$          |
| $\text{Loss}$     | $\boldsymbol{\hat{y}},\boldsymbol{y}$ | $\ell_{\text{BCE}}=-\frac{1}{K}\left[\boldsymbol{y}^{T}\log(\boldsymbol{\hat{y}})+(1-\boldsymbol{y})^{T}\log(1-\boldsymbol{\hat{y}})\right]$ |

In the equations of **c)** the derivative $\frac{\partial \ell}{\partial\boldsymbol{\hat{y}}}$.

In the equations of **d)**, since the derivative $\frac{\partial \ell}{\partial\boldsymbol{\hat{y}}}$ changes, so do its components, let's compute them and write the matrix representation:
$$
\ell_{\text{BCE}}=
-\frac{1}{K}\sum_{j}\left[y_j\log(\hat{y}_j)+(1-y_j)\log(1-\hat{y}_j)\right]. \\

\frac{\partial \ell_{\text{BCE}}}{\partial \hat{y}_i} = 
\frac{1}{K}\frac{\hat{y}_i - y_i}{\hat{y}_i(1-\hat{y}_i)}. \\

\frac{\partial \ell_{\text{BCE}}}{\partial \boldsymbol{\hat{y}}} = 
\frac{1}{K}
\begin{pmatrix}
\frac{\hat{y}_0 - y_0}{\hat{y}_0(1-\hat{y}_0)} &
\frac{\hat{y}_1 - y_1}{\hat{y}_1(1-\hat{y}_1)} &
\ldots &
\frac{\hat{y}_{d_\boldsymbol{\hat{y}}} - y_{d_\boldsymbol{\hat{y}}}}{\hat{y}_{d_\boldsymbol{\hat{y}}}(1-\hat{y}_{d_\boldsymbol{\hat{y}}})}
\end{pmatrix}.
$$

#### Solution c)

Because the the calculation and the calculation of the gradient is faster and $\text{ReLU}$ is good avoiding gradient vanishing.  
