import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

st.set_page_config(page_title="Optimasi Bukan Korupsi", layout="wide")

st.title("Optimasi Bukan Korupsi")
st.subheader("Optimasi Pendapatan Negara dengan Penentuan Kombinasi Kebijakan Pajak dan PNBP menggunakan Metode Grafis 2 Variabel")

st.markdown(
    """ ### Model 
    Maksimumkan: Z = 5X + 4Y 
    dengan kendala: 1. 2X + Y ≤ 20 
                    2. X + 2Y ≤ 20 
                    3. X ≥ 2 
                    4. X ≥ 0, Y ≥ 0 
    """
    )

# linear programming
c = [-5, -4]

A = [
    [2, 1],
    [1, 2]
]

b = [20, 20]

bounds = [(2, None), (0, None)]

result = linprog(
    c=c,
    A_ub=A,
    b_ub=b,
    bounds=bounds,
    method='highs'
)

x_opt, y_opt = result.x
z_opt = -result.fun

st.success("solusi optimal ditemukan!")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Nilai optimal X", f"{x_opt:2f}")

with col2:
    st.metric("Nilai optimal Y", f"{y_opt:2f}")

with col3:
    st.metric("Nilai optimal Z", f"{z_opt:2f}")

#Plotting
x = np.linspace(0, 20, 400) 
y1 = 20 - 2*x 
y2 = (20 - x)/2 
y1 = np.maximum(y1, 0) 
y2 = np.maximum(y2, 0)

fig, ax = plt.subplots(figsize=(10,8))

ax.plot(x, y1, label="2x + y = 20")
ax.plot(x,y2, label="x + 2y = 20")
ax.axvline(2, linestyle='--', label="x = 2")

vertices_x = [2,2,20/3,10]
vertices_y = [0,9,20/3,0]

ax.fill(
    vertices_x,
    vertices_y,
    alpha=0.3,
    label="Daerah Feasible"
)

# titik optimal
ax.scatter(
    x_opt,
    y_opt,
    s=150,
    color="red",
    label="titik optimal"
)

ax.annotate(
    f'({x_opt: .2f}, {y_opt:.2f})',
    (x_opt, y_opt),
    xytext=(8, 8),
    textcoords="offset points"
)

ax.set_xlim(0,20)
ax.set_ylim(0,20)

ax.set_xlabel('Jumlah kebijakan pajak (X)')
ax.set_ylabel('Jumlah kebijakan PNBP (Y)')

ax.set_title('Metode Grafis 2 Variabel')

ax.grid(True)
ax.legend()

st.pyplot(fig)