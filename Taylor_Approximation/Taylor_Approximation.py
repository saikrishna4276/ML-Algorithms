import jax.numpy as jnp
from jax import grad
from math import factorial
import matplotlib.pyplot as plt
import streamlit as st

def fx(x):
    return jnp.cos(x)

def taylor(fun, degree, x, a=0.):
    y = fun(a)
    gradient = grad(fun)
    for i in range(1,degree+1):
        y += (gradient(a)*(x-a)**i)/factorial(i)
        gradient = grad(gradient)
    return y

x_plot = jnp.linspace(0,10,100)
x = jnp.linspace(-jnp.pi, jnp.pi, 100)

# Create a streamlit app
st.title("Taylor Approximation")
st.write(
    "This app show the taylor approximation of a function with varying degrees")

st.write("DEMO: f(x) = sin(x)")
# The sidebar contains the sliders
with st.sidebar:
    # Create a slider for degree
    degree = st.slider("Degree", 1, 15, 1)


approximation = taylor(fx, degree, x)

fig, ax = plt.subplots()
ax.plot(x,fx(x))
ax.plot(x,approximation)
st.pyplot(fig)
