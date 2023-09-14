import streamlit as st
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad
import sympy
# Store the initial value of widgets in session state
from sympy import *
from sympy.abc import *
from sympy.parsing.sympy_parser import parse_expr
from math import factorial
import mpld3
import streamlit.components.v1 as components
st.title("Taylor Approximation")
st.write(
    "This app shows the taylor approximation of a function with varying degrees")
with st.sidebar:
    try:
        z = parse_expr(st.text_input('Give an expession', 'sin(x)'))
        symbols = set(z.free_symbols)
        symbols_tuple = tuple(symbols)
        z = lambdify(symbols_tuple,z,"jax") 
    except:
        st.warning("Invalid expression")
        z = parse_expr('sin(x)')
        symbols = set(z.free_symbols)
        symbols_tuple = tuple(symbols)
        z = lambdify(symbols_tuple,z,"jax")
def f(a):
    global z
    try:
        return z(a)
    except:
        st.warning(f"Function not defined around {a}")
        return jnp.nan
def taylor(fun, degree, x_plot, a=0.):
    y = fun(a)
    #st.write(y)
    gradient = grad(fun)
    for i in range(1,degree+1):
        y += (gradient(a)*(x_plot-a)**i)/factorial(i)
        gradient = grad(gradient)
    y = jnp.nan_to_num(y, nan=0.0, posinf=10e+6, neginf=-10e+6)
    
    return y

with st.sidebar:
    # Create a slider for degree
    degree = st.slider("Degree", 1, 10, 1)
    a = st.text_input("about a","0")
a = float(a)
x_plot = jnp.linspace(a-2,a+2,100)
try:
    z(x_plot)
except:
    st.warning("Invalid expression")
    z = parse_expr('sin(x)')
    symbols = set(z.free_symbols)
    symbols_tuple = tuple(symbols)
    z = lambdify(symbols_tuple,z,"jax")
approximation = taylor(f,degree,x_plot,a)

#st.write(approximation)
x = jnp.linspace(a-1, a+1,100)
fig, ax = plt.subplots()

dataset = st.sidebar.selectbox("Select Plot-view", ("Fixed", "Dynamic"))
if dataset == "Fixed":
    ax.plot(x,approximation,label='Approximated Curve')
    ax.plot(x,z(x_plot),label='Actual curve')
    ax.set_ylim(min(z(x_plot))-2, max(z(x_plot))+2)
    ax.legend()
    st.pyplot(fig)
else:
    ax.plot(x,approximation,label='Approximated Curve')
    ax.plot(x,z(x_plot),label='Actual curve')
    ax.set_ylim(min(z(x_plot))-2, max(z(x_plot))+2)
    ax.legend()
    fig_html = mpld3.fig_to_html(fig, template_type="general" )
    components.html(fig_html, height=800, width=1600)

