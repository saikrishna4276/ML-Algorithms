import streamlit as st
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad
# Store the initial value of widgets in session state
from sympy import *
from sympy.abc import X,x
from sympy.parsing.sympy_parser import parse_expr
from math import factorial
st.set_option('deprecation.showPyplotGlobalUse', False)
if 'count' not in st.session_state:
    st.session_state.count = 0
with st.sidebar:
    z = parse_expr(st.text_input('Give an expession', 'sin(x)'))
    z = lambdify(x,z,"jax")
    if st.button("sin(x)"):
        st.session_state.count+=sin(X)
    if st.button("cos(x)"):
        st.session_state.count+=cos(X)
    if st.button("log(1+x)"):
        st.session_state.count+=log(1+X)
    if st.button("log(1-x)"):
        st.session_state.count+=log(1-X)
    if st.button("exp(x)"):
        st.session_state.count+=exp(X)
    if st.button('RESET'):
        st.session_state.count=0
    st.text(st.session_state.count)
def f(a):
    global z
    return z(a)
def taylor(fun, degree, x_plot, a=0.):
    y = fun(a)
    #st.write(y)
    gradient = grad(fun)
    for i in range(1,degree+1):
        y += (gradient(a)*(x_plot-a)**i)/factorial(i)
        gradient = grad(gradient)
    return y
x_plot = jnp.linspace(-jnp.pi, jnp.pi, 100)
with st.sidebar:
    # Create a slider for degree
    degree = st.slider("Degree", 1, 15, 1)
approximation = taylor(f,degree,x_plot)
#st.write(approximation)
x = jnp.linspace(-3,3,100)
fig, ax = plt.subplots()
ax.plot(x,z(x_plot))
ax.plot(x,approximation)
st.pyplot(fig)