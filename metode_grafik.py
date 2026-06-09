import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from scipy.optimize import linprog
import time

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Optimasi Pendapatan Negara",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0f2c4a 0%, #1a4a7a 60%, #0d3d6b 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    border-left: 5px solid #f5a623;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "∑";
    position: absolute;
    right: 3rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 10rem;
    color: rgba(245,166,35,0.08);
    font-family: 'IBM Plex Mono', monospace;
    pointer-events: none;
}
.hero-label {
    color: #f5a623;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hero-title {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.15;
    margin-bottom: 0.75rem;
}
.hero-subtitle {
    color: #a8c4e0;
    font-size: 0.95rem;
    font-weight: 300;
    line-height: 1.6;
    max-width: 680px;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin: 1.5rem 0; }
.metric-card {
    flex: 1;
    background: #0f2c4a;
    border: 1px solid #1e4d7a;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card.highlight { border-color: #f5a623; background: #16345a; }
.metric-label {
    color: #6fa3cc;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.metric-value {
    color: #f5a623;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
}
.metric-unit { color: #a8c4e0; font-size: 0.8rem; margin-top: 0.2rem; }

/* ── Section headers ── */
.section-head {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2rem 0 1rem 0;
}
.section-number {
    background: #f5a623;
    color: #0f2c4a;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 0.75rem;
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.section-title {
    color: #f5a623;
    font-size: 1.1rem;
    font-weight: 600;
}

/* ── Step cards ── */
.step-card {
    background: #0d2640;
    border: 1px solid #1a3f65;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.step-card-title {
    color: #f5a623;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.step-card p { color: #a8c4e0; font-size: 0.9rem; line-height: 1.7; margin: 0; }
.step-card code {
    background: #1a3f65;
    color: #7dd3fc;
    padding: 0.1em 0.4em;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85em;
}

/* ── Vertex table ── */
.vtable { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
.vtable th {
    background: #1a3f65;
    color: #6fa3cc;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.5rem 0.75rem;
    text-align: center;
    border: 1px solid #1e4d7a;
}
.vtable td {
    color: #dce8f5;
    padding: 0.5rem 0.75rem;
    text-align: center;
    border: 1px solid #1a3f65;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}
.vtable tr.optimal td { background: #1e3d20; color: #86efac; }
.vtable tr:hover td { background: #12304f; }

/* ── Model box ── */
.model-box {
    background: #060f1a;
    border: 1px solid #1e4d7a;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #7dd3fc;
    font-size: 0.9rem;
    line-height: 2;
}
.model-box .obj { color: #f5a623; font-weight: 600; }
.model-box .constraint { color: #a8c4e0; }
.model-box .var { color: #86efac; }

/* ── Overrides ── */
div[data-testid="stMetric"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-label">Linear Programming · Metode Grafis 2 Variabel</div>
  <div class="hero-title">Optimasi Pendapatan Negara</div>
  <div class="hero-subtitle">
    Penentuan kombinasi kebijakan Pajak (X) dan PNBP (Y) yang memaksimalkan 
    pendapatan negara menggunakan program linier dengan visualisasi daerah feasible.
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Model ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
  <div class="section-number">M</div>
  <div class="section-title">Model Program Linier</div>
</div>
<div class="model-box">
  <span class="obj">Maksimumkan:&nbsp; Z = 5X + 4Y</span><br>
  <span style="color:#6fa3cc">Dengan kendala:</span><br>
  <span class="constraint">&nbsp; (1)&nbsp; 2X + Y  ≤  20</span><br>
  <span class="constraint">&nbsp; (2)&nbsp; X + 2Y  ≤  20</span><br>
  <span class="constraint">&nbsp; (3)&nbsp; X  ≥  2</span><br>
  <span class="var">&nbsp; (4)&nbsp; X ≥ 0,&nbsp; Y ≥ 0</span>
</div>
""", unsafe_allow_html=True)

# ─── Solve ───────────────────────────────────────────────────────────────────
c_lp   = [-5, -4]
A_ub   = [[2, 1], [1, 2]]
b_ub   = [20, 20]
bounds = [(2, None), (0, None)]
result = linprog(c=c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
x_opt, y_opt = result.x
z_opt = -result.fun

# ─── Results ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
  <div class="section-number">★</div>
  <div class="section-title">Solusi Optimal</div>
</div>
<div class="metric-row">
  <div class="metric-card">
    <div class="metric-label">Kebijakan Pajak</div>
    <div class="metric-value">{x:.2f}</div>
    <div class="metric-unit">Nilai X optimal</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Kebijakan PNBP</div>
    <div class="metric-value">{y:.2f}</div>
    <div class="metric-unit">Nilai Y optimal</div>
  </div>
  <div class="metric-card highlight">
    <div class="metric-label">Pendapatan Maksimum</div>
    <div class="metric-value">{z:.2f}</div>
    <div class="metric-unit">Nilai Z = 5X + 4Y</div>
  </div>
</div>
""".format(x=x_opt, y=y_opt, z=z_opt), unsafe_allow_html=True)

# ─── Step-by-step ─────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
  <div class="section-number">1</div>
  <div class="section-title">Langkah Penyelesaian</div>
</div>
""", unsafe_allow_html=True)

steps = [
    ("LANGKAH 1 — Identifikasi Variabel", """
        Tentukan dua variabel keputusan:<br>
        &nbsp;• <code>X</code> = jumlah kebijakan pajak yang diterapkan<br>
        &nbsp;• <code>Y</code> = jumlah kebijakan PNBP yang diterapkan<br>
        Tujuan: memaksimalkan <code>Z = 5X + 4Y</code> (total pendapatan negara).
    """),
    ("LANGKAH 2 — Gambar Garis Batas Kendala", """
        Setiap pertidaksamaan diubah menjadi persamaan garis lurus:<br>
        &nbsp;• <code>2X + Y = 20</code> → titik potong: (0, 20) dan (10, 0)<br>
        &nbsp;• <code>X + 2Y = 20</code> → titik potong: (0, 10) dan (20, 0)<br>
        &nbsp;• <code>X = 2</code> → garis vertikal di X = 2<br>
        Ketiga garis membentuk batas daerah feasible.
    """),
    ("LANGKAH 3 — Tentukan Daerah Feasible", """
        Daerah feasible adalah himpunan titik yang memenuhi <em>semua</em> kendala sekaligus.<br>
        Dengan memeriksa arah pertidaksamaan, daerah yang valid adalah sisi yang mengandung 
        nilai-nilai yang memenuhi kendala <code>≤ 20</code> dan <code>X ≥ 2</code>.<br>
        Daerah ini digambar sebagai poligon terarsir pada grafik.
    """),
    ("LANGKAH 4 — Temukan Titik-Titik Sudut (Vertices)", """
        Titik optimal hanya bisa berada di sudut daerah feasible. Hitung perpotongan:<br>
        &nbsp;• <code>A = (2, 0)</code> → X = 2, Y = 0<br>
        &nbsp;• <code>B = (2, 9)</code> → X = 2 & 2X + Y = 20<br>
        &nbsp;• <code>C = (20/3, 20/3) ≈ (6.67, 6.67)</code> → 2X+Y=20 & X+2Y=20<br>
        &nbsp;• <code>D = (10, 0)</code> → X + 2Y = 20 & Y = 0
    """),
    ("LANGKAH 5 — Evaluasi Fungsi Objektif di Setiap Titik Sudut", """
        Substitusikan koordinat setiap sudut ke <code>Z = 5X + 4Y</code>:<br>
        &nbsp;• A (2, 0): Z = 5(2) + 4(0) = <code>10</code><br>
        &nbsp;• B (2, 9): Z = 5(2) + 4(9) = 10 + 36 = <code>46</code><br>
        &nbsp;• C (6.67, 6.67): Z = 5(6.67) + 4(6.67) = 33.33 + 26.67 = <code>60.00</code><br>
        &nbsp;• D (10, 0): Z = 5(10) + 4(0) = <code>50</code><br>
        Nilai Z terbesar ada di titik <strong>C</strong>.
    """),
    ("LANGKAH 6 — Kesimpulan", f"""
        Titik optimal yang memaksimalkan pendapatan negara adalah:<br>
        &nbsp;• <code>X = {x_opt:.2f}</code> (kebijakan pajak)<br>
        &nbsp;• <code>Y = {y_opt:.2f}</code> (kebijakan PNBP)<br>
        &nbsp;• <code>Z = {z_opt:.2f}</code> (pendapatan maksimum)<br>
        Artinya, kombinasi ~6.67 kebijakan pajak dan ~6.67 kebijakan PNBP 
        menghasilkan pendapatan negara optimal.
    """),
]

for title, body in steps:
    st.markdown(f"""
    <div class="step-card">
      <div class="step-card-title">{title}</div>
      <p>{body}</p>
    </div>
    """, unsafe_allow_html=True)

# ─── Vertex table ────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
  <div class="section-number">2</div>
  <div class="section-title">Tabel Evaluasi Titik Sudut</div>
</div>
""", unsafe_allow_html=True)

vertices = [
    ("A", 2,      0,      "2X+Y≤20 & X≥2 & Y=0"),
    ("B", 2,      9,      "X=2 & 2X+Y=20"),
    ("C", 20/3,   20/3,   "2X+Y=20 & X+2Y=20"),
    ("D", 10,     0,      "X+2Y=20 & Y=0"),
]

rows = ""
for label, vx, vy, desc in vertices:
    z = 5*vx + 4*vy
    cls = "optimal" if abs(vx - x_opt) < 0.01 and abs(vy - y_opt) < 0.01 else ""
    mark = " ← OPTIMAL" if cls else ""
    rows += f"""<tr class="{cls}">
      <td>{label}</td>
      <td>({vx:.2f}, {vy:.2f})</td>
      <td>{z:.2f}</td>
      <td style="font-size:0.75rem; color:#0d1e30">{desc}{mark}</td>
    </tr>"""

st.markdown(f"""
<table class="vtable">
  <thead>
    <tr>
      <th>Titik</th><th>Koordinat (X, Y)</th><th>Z = 5X + 4Y</th><th>Asal Perpotongan</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
""", unsafe_allow_html=True)

# ─── Animated plot ───────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
  <div class="section-number">3</div>
  <div class="section-title">Visualisasi Daerah Feasible & Titik Optimal</div>
</div>
""", unsafe_allow_html=True)

animate = st.toggle("▶  Tampilkan animasi langkah-demi-langkah", value=False)

# ─── Color palette matched to CSS ───────────────────────────────────────────
BG       = "#0d1e30"
GRID     = "#1a3f65"
AXES     = "#2a5580"
C1       = "#f5a623"   # constraint 1 — amber
C2       = "#38bdf8"   # constraint 2 — sky
C3       = "#a78bfa"   # constraint 3 — violet
FILL     = "#1d4ed8"   # feasible region
OPT_C    = "#f87171"   # optimal point
VERTEX_C = "#86efac"   # other vertices
LABEL_C  = "#e0eaf5"

def make_figure(show_c1=True, show_c2=True, show_c3=True,
                show_fill=True, show_vertices=True, show_optimal=True,
                show_isoprofit=True):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for spine in ax.spines.values():
        spine.set_edgecolor(AXES)
    ax.tick_params(colors=LABEL_C, labelsize=9)
    ax.xaxis.label.set_color(LABEL_C)
    ax.yaxis.label.set_color(LABEL_C)
    ax.title.set_color(LABEL_C)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle='--', alpha=0.7)

    x = np.linspace(0, 14, 600)

    if show_c1:
        y1 = np.clip(20 - 2*x, 0, None)
        ax.plot(x, y1, color=C1, lw=2, label="2X + Y = 20", zorder=4)
    if show_c2:
        y2 = np.clip((20 - x)/2, 0, None)
        ax.plot(x, y2, color=C2, lw=2, label="X + 2Y = 20", zorder=4)
    if show_c3:
        ax.axvline(2, color=C3, lw=2, linestyle='--', label="X = 2", zorder=4)

    if show_fill:
        vx = [2, 2, 20/3, 10]
        vy = [0, 9, 20/3, 0]
        ax.fill(vx, vy, color=FILL, alpha=0.25, zorder=2, label="Daerah Feasible")
        ax.plot(vx + [vx[0]], vy + [vy[0]], color=FILL, lw=1.5, alpha=0.7, zorder=3)

    if show_vertices:
        for label, vx, vy, _ in vertices:
            if abs(vx - x_opt) < 0.01 and abs(vy - y_opt) < 0.01:
                continue
            ax.scatter(vx, vy, s=80, color=VERTEX_C, zorder=6, edgecolors='white', linewidths=0.8)
            ax.annotate(f' {label}({vx:.1f},{vy:.1f})',
                        (vx, vy), color=VERTEX_C, fontsize=8,
                        xytext=(5, 5), textcoords='offset points')

    if show_isoprofit:
        # isoprofit line through optimal point
        z_line = z_opt
        y_iso = np.clip((z_line - 5*x)/4, 0, 14)
        ax.plot(x, y_iso, color=OPT_C, lw=1.2, linestyle=':', alpha=0.6,
                label=f"Z = {z_line:.0f} (iso-profit)")

    if show_optimal:
        ax.scatter(x_opt, y_opt, s=220, color=OPT_C, zorder=8,
                   edgecolors='white', linewidths=1.5, label="Titik Optimal")
        ax.annotate(f'  OPTIMAL\n  ({x_opt:.2f}, {y_opt:.2f})\n  Z = {z_opt:.2f}',
                    (x_opt, y_opt), color=OPT_C, fontsize=9, fontweight='bold',
                    xytext=(12, 6), textcoords='offset points')

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 13)
    ax.set_xlabel('Kebijakan Pajak (X)', fontsize=10, labelpad=8)
    ax.set_ylabel('Kebijakan PNBP (Y)', fontsize=10, labelpad=8)
    ax.set_title('Metode Grafis — Daerah Feasible & Solusi Optimal', fontsize=11, pad=12)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=LABEL_C, fontsize=8.5,
              framealpha=0.9, loc='upper right')
    fig.tight_layout()
    return fig

if animate:
    placeholder = st.empty()
    btn_col1, btn_col2 = st.columns([1, 5])
    with btn_col1:
        run_btn = st.button("▶ Putar Ulang", type="primary")

    animation_steps = [
        dict(show_c1=True,  show_c2=False, show_c3=False, show_fill=False,
             show_vertices=False, show_optimal=False, show_isoprofit=False),
        dict(show_c1=True,  show_c2=True,  show_c3=False, show_fill=False,
             show_vertices=False, show_optimal=False, show_isoprofit=False),
        dict(show_c1=True,  show_c2=True,  show_c3=True,  show_fill=False,
             show_vertices=False, show_optimal=False, show_isoprofit=False),
        dict(show_c1=True,  show_c2=True,  show_c3=True,  show_fill=True,
             show_vertices=False, show_optimal=False, show_isoprofit=False),
        dict(show_c1=True,  show_c2=True,  show_c3=True,  show_fill=True,
             show_vertices=True,  show_optimal=False, show_isoprofit=False),
        dict(show_c1=True,  show_c2=True,  show_c3=True,  show_fill=True,
             show_vertices=True,  show_optimal=True,  show_isoprofit=True),
    ]
    step_labels = [
        "Menggambar kendala 1: 2X + Y ≤ 20",
        "Menggambar kendala 2: X + 2Y ≤ 20",
        "Menggambar kendala 3: X ≥ 2",
        "Mengarsir daerah feasible",
        "Menandai titik-titik sudut",
        "Menemukan titik optimal ✓",
    ]

    progress_bar = st.progress(0)
    status_text  = st.empty()

    for i, params in enumerate(animation_steps):
        fig = make_figure(**params)
        placeholder.pyplot(fig)
        plt.close(fig)
        progress_bar.progress((i + 1) / len(animation_steps))
        status_text.markdown(
            f"<span style='color:#f5a623; font-family:IBM Plex Mono; font-size:0.85rem'>"
            f"STEP {i+1}/{len(animation_steps)} — {step_labels[i]}</span>",
            unsafe_allow_html=True
        )
        time.sleep(1.4)

    status_text.markdown(
        "<span style='color:#86efac; font-family:IBM Plex Mono; font-size:0.85rem'>"
        "✓ Selesai — Solusi optimal ditemukan!</span>",
        unsafe_allow_html=True
    )

else:
    fig = make_figure()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""<br>""", unsafe_allow_html=True)
st.markdown("""
<div style="border-top:1px solid #1a3f65; padding-top:1rem; color:#3d6a9a; 
            font-family:'IBM Plex Mono',monospace; font-size:0.72rem; text-align:center;">
  Optimasi Bukan Korupsi &nbsp;·&nbsp; Linear Programming &nbsp;·&nbsp; 
  Metode Grafis 2 Variabel &nbsp;·&nbsp; scipy.optimize.linprog / HiGHS solver
</div>
""", unsafe_allow_html=True)