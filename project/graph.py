from dataclasses import dataclass

@dataclass(frozen=True)
class PlotSpec:
    domain: str          # "Q" or "k"
    quantity: str        # "absF", "E", "BC", "phase"
    plot_type: str       # "imshow" | "quiver" | "surface"
    dim: int
    cmap: str = "viridis"
    # LaTeX title (mathtext) and filename templates
    title_tmpl: str = ""
    fname_tmpl: str = ""

class GraphProtocol(Enum):
    #probability/abs
    AbsF2D = auto() 
    ExcProb2D = auto()
    #Energy
    ExcE2D = auto() 
    NiE2D = auto() 
    #Berry curvature
    FBC= auto() 
    ExcBC=auto() 
    NiBC= auto() 
    #Quiver
    FQuiver = auto()
    KQuiver = auto()
    #3D test also multi
    EQ3D = auto() 
    AbsF3D = auto()
    ExcProb3D = auto()
    #1D
    ExcBandQx = auto()
    EQHighSym = auto()
    #for tests
    Test = auto()

PLOT_SPECS = {
    GraphProtocol.AbsF2D: PlotSpec(
        domain="Q", quantity="absF", plot_type="imshow", dim = 2,
        title_tmpl=r"F_{{{band}}}\left(p={px},{py}\right)",
        fname_tmpl="F_{band}_p{p0}_{p1}",
    ),
    GraphProtocol.ExcProb2D: PlotSpec(
        domain="Q", quantity="ExcProb", plot_type="imshow", dim = 2,
        title_tmpl=r"|U_{{{band}}}\left(Q\right)|",
        fname_tmpl="U_{band}_prob2D",
    ),
    GraphProtocol.ExcE2D: PlotSpec(
        domain="Q", quantity="E", plot_type="imshow", dim =2,
        title_tmpl=r"E_{{{band}}}(Q)",
        fname_tmpl="E_{band}_Q",
    ),
    GraphProtocol.NiE2D: PlotSpec(
        domain="k", quantity="E", plot_type="imshow", dim = 2,
        title_tmpl=r"E_{{{band}}}(k)",
        fname_tmpl="E_{band}_k",
    ),
    GraphProtocol.FBC: PlotSpec(
        domain="Q", quantity="BC", plot_type="imshow", dim = 2,
        title_tmpl=r"F_{{{band}}}\ \mathrm{{Berry\ Curvature}}\ \left(p={px},{py}\right)\ C={C}",
        fname_tmpl="F_BC_{band}_p{p0}_{p1}_C{C}",
    ),
    GraphProtocol.ExcBC: PlotSpec(
        domain="Q", quantity="BC", plot_type="imshow", dim = 2,
        title_tmpl=r"Exciton_{{{band}}}\ \mathrm{{Berry\ Curvature}}\ C={C}",
        fname_tmpl="Exc_BC_{band}_C{C}",
    ),
    GraphProtocol.NiBC: PlotSpec(
        domain="k", quantity="BC", plot_type="imshow", dim = 2,
        title_tmpl=r"ni\ n ={band}\ \mathrm{{Berry\ Curvature}}\ C={C}",
        fname_tmpl="ni_BC{band}_C{C}",
    ),
    GraphProtocol.FQuiver: PlotSpec(
        domain="Q", quantity="phase", plot_type="quiver", cmap="hsv", dim = 2,
        title_tmpl=r"F_{{{band}}}\ \left(p={px},{py}\right) \mathrm{{phase}}",
        fname_tmpl="F_{band}_phase_Q_quiver_p{p0}_{p1}",
    ),
    GraphProtocol.KQuiver: PlotSpec(
        domain="k", quantity="phase", plot_type="quiver", cmap="hsv", dim = 2,
        title_tmpl=r"u_{{{band}}}\ \mathrm{{phase}}",
        fname_tmpl="u_{band}_phase_k_quiver",
    ),
    GraphProtocol.EQ3D: PlotSpec(
        domain="Q", quantity="E", plot_type="imshow", dim = 3,
        title_tmpl=r"E_{{{band}}}(Q)",
        fname_tmpl="E_[{band}]_Q_3D",
    ),
    GraphProtocol.AbsF3D: PlotSpec(
        domain="Q", quantity="absF", plot_type="imshow", dim = 3,
        title_tmpl=r"F_{{{band}}}\left(p={px},{py}\right)",
        fname_tmpl="F_{band}_p{p0}_{p1}",
    ),
    GraphProtocol.ExcProb3D: PlotSpec(
        domain="Q", quantity="Probability", plot_type="imshow", dim = 3,
        title_tmpl=r"U_{{{band}}}\ Probability",
        fname_tmpl="U_{band}_prob_3D",
    ),
    GraphProtocol.EQHighSym: PlotSpec(
        domain="Q", quantity="Energy", plot_type="imshow", dim = 1,
        title_tmpl=r"E(Q)",
        fname_tmpl="E_HighSym",
    ),
    GraphProtocol.ExcBandQx: PlotSpec(
        domain="Q", quantity="Energy", plot_type="imshow", dim = 1,
        title_tmpl=r"E(Q)",
        fname_tmpl="E_Q1Const",
    ),
    #--Test--#
    GraphProtocol.Test: PlotSpec(
        domain="Q", quantity="analytic|F|", plot_type="imshow", dim = 2,
        title_tmpl=r"analytic|F|\left(p={px},{py}\right)",
        fname_tmpl="analytic|F|_p{p0}_{p1}",
    ),
}

class InteractiveWindingExplorer:
    """
    Interactive click-to-select-minimum and inspect phase winding.

    Usage
    -----
    explorer = InteractiveWindingExplorer(F, p=(0, 1), radii=[2, 3, 4, 5], search_radius=5)
    """
    def __init__(self, F, radii=(2, 3, 4, 5), search_radius=5,p=['','']):
        self.F = F
        self.radii = list(radii)
        self.search_radius = search_radius
        
        self.A = np.abs(self.F)

        self.fig, (self.ax_img, self.ax_txt) = plt.subplots(
            1, 2, figsize=(10, 5), constrained_layout=True
        )

        self.im = self.ax_img.imshow(self.A, origin="lower", cmap="viridis")
        self.ax_img.set_title(f"|F[:, :, {p[0]}, {p[1]}]|")
        self.ax_img.set_xlabel("x")
        self.ax_img.set_ylabel("y")
        self.fig.colorbar(self.im, ax=self.ax_img, label="|F|")

        self.marker, = self.ax_img.plot([], [], "ro", ms=8)
        self.loop_artists = []

        self.ax_txt.set_axis_off()
        self.text = self.ax_txt.text(
            0.02, 0.98, "Click near a vortex/core",
            va="top", ha="left", family="monospace"
        )

        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)

    def clear_loops(self):
        for artist in self.loop_artists:
            artist.remove()
        self.loop_artists = []

    def draw_square(self, center, radius, **kwargs):
        iy, ix = center
        x0 = ix - radius
        y0 = iy - radius
        w = 2 * radius
        rect = plt.Rectangle((x0, y0), w, w, fill=False, **kwargs)
        self.ax_img.add_patch(rect)
        self.loop_artists.append(rect)

    def onclick(self, event):
        if event.inaxes != self.ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return

        iy, ix = find_local_min_near_click(
            self.A, event.xdata, event.ydata, search_radius=self.search_radius
        )

        self.marker.set_data([ix], [iy])
        self.clear_loops()

        lines = [
            f"selected minimum: (iy, ix) = ({iy}, {ix})",
            f"|F| there = {self.A[iy, ix]:.6g}",
            "",
            "radius   total_phase      winding     rounded"
        ]

        for r in self.radii:
            total, winding, pts = phase_winding_square(self.F, (iy, ix), r)
            if pts is not None:
                self.draw_square((iy, ix), r, lw=1.5)

            lines.append(
                f"{r:>5d}   "
                f"{total:>12.6f}   "
                f"{winding:>10.6f}   "
                f"{('nan' if np.isnan(winding) else int(np.round(winding))):>7}"
            )

        self.text.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()