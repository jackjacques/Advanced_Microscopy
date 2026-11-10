import matplotlib.pyplot as plt
import numpy as np
import cv2


class Microscopy:

    def __init__(
        self,
        n_cells=10,
        cell_radius_mean=50e-6,
        cell_radius_sigma=10e-6,
        nucleus_radius_frac=0.10,
        delta_n=0.02,
        lambda_ref=550e-9,
        N=4096,
        L=1.0e-3,
        NA=0.1,
        nucleus_amplitude=0.95,
    ):
        self.n_cells = n_cells
        self.cell_radius_mean = cell_radius_mean
        self.cell_radius_sigma = cell_radius_sigma
        self.nucleus_radius_frac = nucleus_radius_frac
        self.delta_n = delta_n
        self.lambda_ref = lambda_ref
        self.N = N
        self.L = L
        self.dx = L / N
        self.NA = NA
        self.nucleus_amplitude = nucleus_amplitude

        self.X, self.Y = self.makeGrid(self.N, self.dx)

        self.phase_map, self.intensity_map = self.makeObject(
            n_cells=self.n_cells,
            cell_radius_mean=self.cell_radius_mean,
            cell_radius_sigma=self.cell_radius_sigma,
            nucleus_radius_frac=self.nucleus_radius_frac,
            delta_n=self.delta_n,
            lambda_ref=self.lambda_ref,
            L=self.L,
            N=self.N,
            nucleus_amplitude=self.nucleus_amplitude,
        )

        self.pupil = self.computePupil(self.NA, self.lambda_ref, self.L, self.N)

    @staticmethod
    def makeGrid(N, dx):
        """Grille centrée x = (-N/2 ... N/2-1) * dx, cohérente avec np.fft.fftfreq(N, dx)."""
        x = (np.arange(N) - N // 2) * dx
        return np.meshgrid(x, x)

    @staticmethod
    def makeObject(
        n_cells=10,
        cell_radius_mean=50e-6,
        cell_radius_sigma=10e-6,
        nucleus_radius_frac=0.10,
        delta_n=0.02,
        lambda_ref=550e-9,
        L=1.0e-3,
        N=4096,
        nucleus_amplitude=0.95,
    ):
        """
        Objet synthétique : cellules transparentes (objets de phase) avec un noyau
        légèrement absorbant (d'après makeObject.m fourni par l'enseignant).

        Retourne (phase_map en rad, intensity_map = amplitude**2 dans [0, 1]).
        """
        dx = L / N
        x = (np.arange(N) - N // 2) * dx

        phase_map = np.zeros((N, N))
        amp_map = np.ones((N, N))

        for _ in range(n_cells):
            # Centre aléatoire dans 80 % du champ de vision
            cx = (np.random.rand() - 0.5) * L * 0.8
            cy = (np.random.rand() - 0.5) * L * 0.8

            # Rayon cellulaire aléatoire (gaussien), borné pour rester physique
            cell_radius = max(
                1e-6, cell_radius_mean + cell_radius_sigma * np.random.randn()
            )
            nucleus_radius = nucleus_radius_frac * cell_radius

            # Décalage aléatoire du noyau à l'intérieur de la cellule
            nuc_dx = (np.random.rand() - 0.5) * cell_radius * 0.5
            nuc_dy = (np.random.rand() - 0.5) * cell_radius * 0.5

            # Calcul limité à la boîte englobant la cellule (le noyau est à
            # l'intérieur) : même résultat qu'en plein champ, beaucoup plus rapide.
            cols = np.flatnonzero(np.abs(x - cx) <= cell_radius)
            rows = np.flatnonzero(np.abs(x - cy) <= cell_radius)
            if cols.size == 0 or rows.size == 0:
                continue
            win = np.ix_(rows, cols)
            Xw, Yw = np.meshgrid(x[cols], x[rows])

            R_cell = np.sqrt((Xw - cx) ** 2 + (Yw - cy) ** 2)
            R_nuc = np.sqrt((Xw - (cx + nuc_dx)) ** 2 + (Yw - (cy + nuc_dy)) ** 2)

            # Épaisseur de la cellule (approximation sphère / lentille)
            thickness = np.zeros_like(Xw)
            inside = R_cell <= cell_radius
            thickness[inside] = 2 * np.sqrt(cell_radius**2 - R_cell[inside] ** 2)

            # Phase induite : 2*pi * delta_n * épaisseur / lambda
            phi_cell = 2 * np.pi * delta_n * thickness / lambda_ref

            # Absorption uniquement dans le noyau
            amp_cell = np.ones_like(Xw)
            amp_cell[R_nuc < nucleus_radius] = nucleus_amplitude

            # Les phases s'additionnent, les transmissions se multiplient
            phase_map[win] += phi_cell
            amp_map[win] *= amp_cell

        return phase_map, amp_map**2


    # ================= PART 2: Resolution and the Pupil Function (Q2, Q3) =================

    @staticmethod
    def computePupil(NA, lambda_ref, L, N):
        """Masque binaire circulaire dans l'espace de Fourier."""
        fc = NA / lambda_ref
        dx = L / N

        fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
        fy = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
        FX, FY = np.meshgrid(fx, fy)

        pupil = (FX**2 + FY**2 <= fc**2).astype(float)
        return pupil

    def get_complex_field(self):
        """E_in = sqrt(I) * exp(i * phi)"""
        return np.sqrt(self.intensity_map) * np.exp(1j * self.phase_map)

    def applyPupilImaging(self, NA):
        """Simule le passage dans le système 4f pour une NA donnée."""
        E_in = self.get_complex_field()

        E_fourier = np.fft.fftshift(np.fft.fft2(E_in))

        pupil = self.computePupil(NA, self.lambda_ref, self.L, self.N)
        E_filtered = E_fourier * pupil

        E_out = np.fft.ifft2(np.fft.ifftshift(E_filtered))

        detected_intensity = np.abs(E_out) ** 2
        return detected_intensity

    def nucleusCentre(self, search_radius=20e-6, intensity_map=None):
        """Centre (x, y) [m] du noyau le plus absorbant : barycentre des pixels
        absorbants autour du minimum de transmission (argmin seul donne un bord)."""
        I = self.intensity_map if intensity_map is None else intensity_map
        iy, ix = np.unravel_index(np.argmin(I), I.shape)
        x0, y0 = self.X[iy, ix], self.Y[iy, ix]
        mask = (I < 1) & (
            (self.X - x0) ** 2 + (self.Y - y0) ** 2 < search_radius**2)
        return self.X[mask].mean(), self.Y[mask].mean()

    def plotPupils(self, NAs=(0.01, 0.1, 1.0),
                   save_path="figures/part2_pupil_and_cutoffs.png"):
        """Q2 - Pupilles binaires et coupes 1D pour chaque NA."""
        fig, axes = plt.subplots(2, len(NAs), figsize=(14, 8), constrained_layout=True)
        fx_cpmm = np.fft.fftshift(np.fft.fftfreq(self.N, d=self.dx)) * 1e-3
        extent = [fx_cpmm[0], fx_cpmm[-1], fx_cpmm[0], fx_cpmm[-1]]
        centre = self.N // 2

        for col, na in enumerate(NAs):
            pupil = self.computePupil(na, self.lambda_ref, self.L, self.N)
            fc_cpmm = (na / self.lambda_ref) * 1e-3

            # Ligne 1 : pupille 2D
            im = axes[0, col].imshow(
                pupil, cmap="gray", extent=extent, origin="lower", vmin=0, vmax=1
            )
            axes[0, col].set_title(f"2D pupil (NA = {na})\n$f_c = {fc_cpmm:.1f}$ cycles/mm")
            axes[0, col].set_xlabel(r"$f_x\ (\mathrm{cycles/mm})$")
            axes[0, col].set_ylabel(r"$f_y\ (\mathrm{cycles/mm})$")
            fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

            # Ligne 2 : coupe 1D
            axes[1, col].plot(fx_cpmm, pupil[centre, :], color="crimson", lw=2, label="Pupil")
            axes[1, col].axvline(fc_cpmm, color="black", linestyle="--", alpha=0.7,
                                 label=f"$\\pm f_c$ ({fc_cpmm:.0f})")
            axes[1, col].axvline(-fc_cpmm, color="black", linestyle="--", alpha=0.7)
            axes[1, col].set_title(f"1D cut ($f_y = 0$), NA = {na}")
            axes[1, col].set_xlabel(r"$f_x\ (\mathrm{cycles/mm})$")
            axes[1, col].set_ylabel("Transmission")
            axes[1, col].set_ylim(-0.05, 1.1)
            axes[1, col].grid(True, linestyle=":", alpha=0.6)
            axes[1, col].legend(loc="upper right", fontsize=8)

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def compareNAs(self, NAs=(0.01, 0.1, 1.0), zoom_half_um=60,
                   save_path="figures/part2_cells_vs_NA.png"):
        """Q3 - Image de l'objet pour plusieurs NA (champ complet + zoom sur un noyau)."""
        fov_um = self.L * 1e6
        extent = [-fov_um / 2, fov_um / 2, -fov_um / 2, fov_um / 2]

        # Zoom centré sur le noyau le plus absorbant
        xc, yc = (c * 1e6 for c in self.nucleusCentre())

        fig, axes = plt.subplots(2, len(NAs), figsize=(5 * len(NAs), 9),
                                 constrained_layout=True)
        for col, na in enumerate(NAs):
            image = self.applyPupilImaging(na)
            r_rayleigh = 0.61 * self.lambda_ref / na * 1e6
            for row in range(2):
                ax = axes[row, col]
                im = ax.imshow(image, cmap="gray", extent=extent, origin="lower")
                ax.set_xlabel(r"$x\ (\mu\mathrm{m})$")
                ax.set_ylabel(r"$y\ (\mu\mathrm{m})$")
                if row == 0:
                    ax.set_title(f"NA = {na}  (Rayleigh: {r_rayleigh:.2f} µm)")
                else:
                    ax.set_xlim(xc - zoom_half_um, xc + zoom_half_um)
                    ax.set_ylim(yc - zoom_half_um, yc + zoom_half_um)
                    ax.set_title(f"Zoom on a nucleus (NA = {na})")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Intensity")

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


    def plot_obj(self, figsize=(11, 4.5), save_path="figures/part1_Intensity_and_Phase_maps.png"):
        fov_um = self.L * 1e6
        extent = [-fov_um / 2, fov_um / 2, -fov_um / 2, fov_um / 2]

        fig, (ax_int, ax_phase) = plt.subplots(
            1, 2, figsize=figsize, constrained_layout=True
        )

        im_int = ax_int.imshow(
            self.intensity_map, cmap="gray", extent=extent, origin="lower",
            vmin=self.intensity_map.min(), vmax=1,
        )
        ax_int.set_title(r"Transmitted intensity $I(x, y)$")
        ax_int.set_xlabel(r"$x\ (\mu\mathrm{m})$")
        ax_int.set_ylabel(r"$y\ (\mu\mathrm{m})$")
        fig.colorbar(im_int, ax=ax_int, fraction=0.046, pad=0.04).set_label("Transmission")

        im_phase = ax_phase.imshow(
            self.phase_map, cmap="viridis", extent=extent, origin="lower"
        )
        ax_phase.set_title(r"Phase map $\Delta\phi(x, y)$")
        ax_phase.set_xlabel(r"$x\ (\mu\mathrm{m})$")
        ax_phase.set_ylabel(r"$y\ (\mu\mathrm{m})$")
        fig.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.04).set_label("Phase (rad)")

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    @staticmethod
    def usafLineWidth(group, element):
        """Largeur d'une barre de la mire USAF 1951 (m) : 1 / (2 * 2^(g + (e-1)/6)) mm."""
        lp_per_mm = 2.0 ** (group + (element - 1) / 6)
        return 1e-3 / (2 * lp_per_mm)

    def makeUSAFTarget(self, groups=range(4, 10), margin=10e-6):
        """
        Mire USAF 1951 synthétique à l'échelle réelle (transmission en amplitude :
        barres opaques = 0, fond transparent = 1). Chaque élément = 3 barres
        verticales de largeur w, espacées de w, de longueur 5w.
        Une ligne par groupe, les éléments 1 à 6 de gauche à droite.

        Retourne (amplitude, liste d'éléments {group, element, lp_mm, x0, y0, w}).
        """
        x = (np.arange(self.N) - self.N // 2) * self.dx
        target = np.ones((self.N, self.N))
        elements = []

        y_top = self.L / 2 - margin
        for g in groups:
            w_max = self.usafLineWidth(g, 1)
            y0 = y_top - 5 * w_max  # bas de la ligne du groupe
            x_left = -self.L / 2 + margin
            for e in range(1, 7):
                w = self.usafLineWidth(g, e)
                for b in range(3):
                    xb = x_left + 2 * b * w
                    cols = (x >= xb) & (x < xb + w)
                    rows = (x >= y0) & (x < y0 + 5 * w)
                    target[np.ix_(rows, cols)] = 0.0
                elements.append(dict(group=g, element=e, lp_mm=1e-3 / (2 * w),
                                     x0=x_left, y0=y0, w=w))
                x_left += 5 * w + w_max
                if x_left > self.L / 2:
                    raise ValueError("La mire ne tient pas dans le champ : réduire les groupes.")
            y_top = y0 - 2 * w_max
            if y_top < -self.L / 2:
                raise ValueError("La mire ne tient pas dans le champ : réduire les groupes.")
        return target, elements

    def elementContrast(self, image, el):
        """
        Contraste de Michelson d'un élément : profil horizontal moyenné sur la
        partie centrale des barres, intensité au centre des 2 interstices vs
        au centre des 3 barres.
        """
        x = (np.arange(self.N) - self.N // 2) * self.dx
        w, x0, y0 = el["w"], el["x0"], el["y0"]
        rows = (x >= y0 + 1.5 * w) & (x < y0 + 3.5 * w)
        profile = image[rows, :].mean(axis=0)
        sample_at = lambda xs: np.interp(xs, x, profile)
        bars = sample_at(x0 + w * np.array([0.5, 2.5, 4.5])).mean()
        gaps = sample_at(x0 + w * np.array([1.5, 3.5])).mean()
        return (gaps - bars) / max(gaps + bars, 1e-12)

    def resolution(self, NAs=(0.01, 0.1, 1.0), groups=range(4, 10), threshold=0.15,
                   save_path="figures/part2_Resolution_calibration_USAF1951.png"):
        """
        Q3.3 - Mesure de la résolution avec une mire USAF 1951 synthétique.
        Pour chaque NA : contraste de chaque élément, puis élément le plus fin
        tel que lui et tous les éléments plus grossiers dépassent `threshold`
        (~ critère de Rayleigh : creux de 26 % -> C ~ 0.15).
        Théorie (imagerie cohérente) : fréquence de coupure f_c = NA / lambda.
        """
        target, elements = self.makeUSAFTarget(groups)
        lp = np.array([el["lp_mm"] for el in elements])
        E_fft = np.fft.fftshift(np.fft.fft2(target.astype(complex)))

        fov_um = self.L * 1e6
        extent = [-fov_um / 2, fov_um / 2, -fov_um / 2, fov_um / 2]
        nyquist_lp = 1e-3 / (2 * self.dx)

        fig = plt.figure(figsize=(5 * (len(NAs) + 1), 10), constrained_layout=True)
        gs = fig.add_gridspec(2, len(NAs) + 1)
        ax_t = fig.add_subplot(gs[0, 0])
        ax_t.imshow(target, cmap="gray", extent=extent, origin="lower", vmin=0, vmax=1)
        ax_t.set_title(f"Synthetic USAF 1951 target\n(groups {groups[0]}–{groups[-1]})")
        ax_t.set_xlabel(r"$x\ (\mu\mathrm{m})$")
        ax_t.set_ylabel(r"$y\ (\mu\mathrm{m})$")
        ax_c = fig.add_subplot(gs[1, :])

        print("=== USAF 1951 RESOLUTION ===")
        for col, na in enumerate(NAs, start=1):
            pupil = self.computePupil(na, self.lambda_ref, self.L, self.N)
            img = np.abs(np.fft.ifft2(np.fft.ifftshift(E_fft * pupil))) ** 2
            C = np.array([self.elementContrast(img, el) for el in elements])

            failed = np.flatnonzero(C < threshold)
            last = (failed[0] if failed.size else len(elements)) - 1
            fc_lp = na / self.lambda_ref * 1e-3

            if last >= 0:
                el = elements[last]
                res_txt = (f"G{el['group']} E{el['element']}: {el['lp_mm']:.0f} lp/mm "
                           f"(line width {el['w']*1e6:.2f} µm)")
            else:
                res_txt = f"coarser than G{groups[0]} E1"
            if last == len(elements) - 1:
                res_txt += " = finest element of the target"
            print(f"NA = {na}: f_c = {fc_lp:.0f} lp/mm (period {1e3/fc_lp:.2f} µm) "
                  f"-> resolved up to {res_txt}")

            ax = fig.add_subplot(gs[0, col])
            ax.imshow(img, cmap="gray", extent=extent, origin="lower")
            ax.set_title(f"NA = {na}\nresolved: {res_txt.split(' (')[0]}")
            ax.set_xlabel(r"$x\ (\mu\mathrm{m})$")

            line, = ax_c.semilogx(lp, C, "o-", ms=3, label=f"NA = {na}")
            ax_c.axvline(fc_lp, color=line.get_color(), ls="--", alpha=0.7)

        ax_c.axhline(threshold, color="black", ls=":", label=f"threshold C = {threshold}")
        ax_c.axvline(nyquist_lp, color="gray", ls="-.", label=f"grid Nyquist ({nyquist_lp:.0f} lp/mm)")
        ax_c.set_xlabel("Spatial frequency of the element (lp/mm)")
        ax_c.set_ylabel("Michelson contrast C")
        ax_c.set_title(r"Contrast per USAF element (dashed: coherent cutoff $f_c = NA/\lambda$)")
        ax_c.grid(True, which="both", ls=":", alpha=0.5)
        ax_c.legend()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    # ================= PART 3: Imaging contrast mechanisms (Q4, Q5) =================

    def frequencyRadius(self):
        """Distance to the center in frequency space [1/m] (0 = straight light)."""
        fx = np.fft.fftshift(np.fft.fftfreq(self.N, d=self.dx))
        FX, FY = np.meshgrid(fx, fx)
        return np.sqrt(FX**2 + FY**2)

    def darkfieldPupil(self, NA_in, NA_out):
        """Q4 - Darkfield: ring-shaped (annular) pupil.
        The center (straight, unscattered light) is blocked: only the light
        deflected by the cells, between NA_in and NA_out, reaches the camera."""
        f = self.frequencyRadius()
        f_in = NA_in / self.lambda_ref
        f_out = NA_out / self.lambda_ref
        return ((f > f_in) & (f <= f_out)).astype(float)

    def phaseContrastPupil(self, NA):
        """Q5 - Zernike phase contrast: normal disk pupil, but the straight light
        (zero frequency) is shifted by a quarter wave (multiplied by i)."""
        pupil = self.computePupil(NA, self.lambda_ref, self.L, self.N).astype(complex)
        pupil[self.frequencyRadius() == 0] = 1j
        return pupil

    def imageWithPupil(self, pupil, field=None):
        """Same as applyPupilImaging, but with any pupil and any input field:
        FFT -> x pupil -> inverse FFT -> intensity."""
        if field is None:
            field = self.get_complex_field()
        E_fourier = np.fft.fftshift(np.fft.fft2(field))
        E_out = np.fft.ifft2(np.fft.ifftshift(E_fourier * pupil))
        return np.abs(E_out) ** 2

    def contrastModes(self, NA=0.5, NA_in=0.05, phi_max_weak=0.3,
                      save_path="figures/part3_contrast_modes.png"):
        """
        Q4 + Q5 : brightfield, darkfield et contraste de phase.
        Ligne 1 : objet réel (phase jusqu'à ~30 rad).
        Ligne 2 : objet faible (phase réduite à phi_max_weak) + profil 1D.
        Le champ est toujours simulé exactement (exp(i*phi)) ; la linéarisation
        1 + i*phi ne sert qu'à l'interprétation théorique.
        """
        P = self.computePupil(NA, self.lambda_ref, self.L, self.N)
        P_dark = self.darkfieldPupil(NA_in, NA)
        P_phc = self.phaseContrastPupil(NA)

        # Ligne 1 : objet réel
        bright = self.imageWithPupil(P)
        dark = self.imageWithPupil(P_dark)
        phc = self.imageWithPupil(P_phc)

        # Ligne 2 : objet faible (phi << 1), champ exact
        phi_weak = self.phase_map / self.phase_map.max() * phi_max_weak
        field_weak = np.sqrt(self.intensity_map) * np.exp(1j * phi_weak)
        bright_weak = self.imageWithPupil(P, field_weak)
        phc_weak = self.imageWithPupil(P_phc, field_weak)

        fov_um = self.L * 1e6
        extent = [-fov_um / 2, fov_um / 2, -fov_um / 2, fov_um / 2]

        fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
        panels = [
            (axes[0, 0], bright, f"Brightfield, real object (NA = {NA})"),
            (axes[0, 1], dark, f"Darkfield, real object ({NA_in} < NA ≤ {NA})"),
            (axes[0, 2], phc, f"Phase contrast, real object (NA = {NA})"),
            (axes[1, 0], bright_weak,
             rf"Brightfield, weak object ($\varphi_{{max}}$ = {phi_max_weak} rad)"),
            (axes[1, 1], phc_weak,
             rf"Phase contrast, weak object ($\varphi_{{max}}$ = {phi_max_weak} rad)"),
        ]
        for ax, img, title in panels:
            im = ax.imshow(img, cmap="gray", extent=extent, origin="lower")
            ax.set_title(title)
            ax.set_xlabel(r"$x\ (\mu\mathrm{m})$")
            ax.set_ylabel(r"$y\ (\mu\mathrm{m})$")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Intensity")

        # Profil 1D à travers la cellule de phase maximale
        iy, ix = np.unravel_index(np.argmax(self.phase_map), self.phase_map.shape)
        x_um = self.X[iy, :] * 1e6
        half = 1.5 * self.cell_radius_mean * 1e6
        sel = np.abs(x_um - x_um[ix]) <= half
        for ax in (axes[1, 0], axes[1, 1]):
            ax.axhline(self.Y[iy, 0] * 1e6, color="crimson", lw=0.8, ls="--")
        ax_p = axes[1, 2]
        ax_p.plot(x_um[sel], bright_weak[iy, sel], label="Brightfield")
        ax_p.plot(x_um[sel], phc_weak[iy, sel], label="Phase contrast")
        ax_p.plot(x_um[sel], (1 + phi_weak[iy, sel]) ** 2, "k--", lw=1,
                  label=r"Theory $(1+\varphi)^2$")
        ax_p.set_title("Weak object: profile along the dashed line")
        ax_p.set_xlabel(r"$x\ (\mu\mathrm{m})$")
        ax_p.set_ylabel("Intensity")
        ax_p.grid(True, ls=":", alpha=0.6)
        ax_p.legend()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    # ================= PART 4: Detector sampling and noise (Q6) =================

    def resampleToCamera(self, image, pitch_obj):
        """Intègre l'image sur des pixels de taille pitch_obj (plan objet).
        INTER_AREA = moyenne sur la surface de chaque pixel, comme un vrai pixel."""
        n_pix = int(round(self.L / pitch_obj))
        return cv2.resize(image.astype(np.float64), (n_pix, n_pix),
                          interpolation=cv2.INTER_AREA)

    def addCameraEffects(self, image, pixel_pitch, snr, magnification=1.0, seed=None):
        """Q6 - Simule une caméra : (1) intégration de la lumière par pixel,
        (2) bruit de photons (Poisson).

        pixel_pitch   : pas du pixel de la caméra [m]
        snr           : SNR sur le fond de l'image : le fond reçoit snr^2 photons
                        (Poisson : SNR = N / sqrt(N) = sqrt(N))
        magnification : grossissement M. Dans le plan objet, un pixel
                        correspond à pixel_pitch / M.

        Retourne l'image bruitée normalisée par le fond (fond ~ 1).
        """
        # 1. Rééchantillonnage : taille du pixel ramenée dans le plan objet
        pitch_obj = pixel_pitch / magnification
        if pitch_obj < self.dx:
            print(f"Warning: object-plane pixel ({pitch_obj*1e6:.2f} µm) < grid dx "
                  f"({self.dx*1e6:.2f} µm): oversampling, no extra information. "
                  f"Increase N.")
        resampled = np.clip(self.resampleToCamera(image, pitch_obj), 0, None)

        # 2. Bruit de Poisson. Normalisation par le fond (médiane) et non par le
        # maximum : le maximum est une frange de bord très fine, que les gros
        # pixels moyennent, ce qui rendrait le nombre de photons dépendant du pixel.
        background = max(np.median(resampled), 1e-12)
        photons_bg = snr**2
        rng = np.random.default_rng(seed)
        noisy = rng.poisson(resampled / background * photons_bg)

        # 3. Échelle relative au fond pour l'affichage
        return noisy / photons_bg

    def cameraComparison(self, NA=0.5, cameras=((1e-6, 30), (3.5e-6, 60)),
                         magnification=1.0, half_window_um=30,
                         save_path="figures/part4_camera.png"):
        """Q6 - Même objet, deux caméras : (pixel 1 µm, SNR 30) vs (pixel 3.5 µm, SNR 60).
        Ligne 1 : champ complet. Ligne 2 : zoom sur un noyau.
        Ligne 3 : profil horizontal à travers le noyau.
        Affiche aussi le CNR du noyau : (I_fond - I_noyau) / sigma_fond."""
        ideal = self.applyPupilImaging(NA)
        ideal = ideal / np.median(ideal)  # fond = 1

        # Noyau le plus absorbant : centre du zoom et ligne du profil
        xc, yc = self.nucleusCentre()

        # Masques "cellule" et "ce noyau" sur la grille fine (pour le CNR)
        cell_mask = (self.phase_map > 0).astype(float)
        near = (self.X - xc) ** 2 + (self.Y - yc) ** 2 < (half_window_um * 1e-6 / 2) ** 2
        nuc_mask = ((self.intensity_map < 1) & near).astype(float)

        nyquist_um = self.lambda_ref / (4 * NA) * 1e6
        nuc_diam_um = 2 * self.nucleus_radius_frac * self.cell_radius_mean * 1e6
        print(f"=== CAMERAS (NA = {NA}, M = {magnification}) ===")
        print(f"Nyquist for this NA: object pixel <= lambda/(4NA) = {nyquist_um:.2f} µm")

        entries = [(ideal, self.dx, f"Ideal image\n(dx = {self.dx*1e6:.2f} µm, no noise)")]
        for pitch, snr in cameras:
            p_obj = pitch / magnification
            img = self.addCameraEffects(ideal, pitch, snr, magnification, seed=0)

            # CNR du noyau : pixels hors cellules = fond, pixels couverts à >50 % = noyau
            bg = self.resampleToCamera(cell_mask, p_obj) == 0
            nuc_cov = self.resampleToCamera(nuc_mask, p_obj)
            nuc = nuc_cov > 0.5
            if not nuc.any():
                nuc = nuc_cov == nuc_cov.max()
            I_bg, sigma_bg = img[bg].mean(), img[bg].std()
            cnr = (I_bg - img[nuc].mean()) / sigma_bg
            print(f"Pixel {pitch*1e6:.1f} µm, SNR {snr}: object pixel = {p_obj*1e6:.2f} µm, "
                  f"{snr**2} photons on background, nucleus ~ {nuc_diam_um/(p_obj*1e6):.1f} px wide, "
                  f"background noise = {sigma_bg*100:.1f} %, nucleus CNR per pixel = {cnr:.1f} "
                  f"({nuc.sum()} px), CNR x sqrt(n_px) = {cnr*np.sqrt(nuc.sum()):.1f}")
            entries.append((img, p_obj, f"Pixel {pitch*1e6:.1f} µm, SNR = {snr}\n"
                                         f"nucleus CNR/pixel = {cnr:.1f}"))

        fov_um = self.L * 1e6
        extent = [-fov_um / 2, fov_um / 2, -fov_um / 2, fov_um / 2]
        vmin, vmax = 0.75, 1.25

        fig = plt.figure(figsize=(5 * len(entries), 13), constrained_layout=True)
        gs = fig.add_gridspec(3, len(entries), height_ratios=[1, 1, 0.7])
        ax_p = fig.add_subplot(gs[2, :])
        for col, (img, p_obj, title) in enumerate(entries):
            for row in range(2):
                ax = fig.add_subplot(gs[row, col])
                im = ax.imshow(img, cmap="gray", extent=extent, origin="lower",
                               interpolation="nearest", vmin=vmin, vmax=vmax)
                ax.set_xlabel(r"$x\ (\mu\mathrm{m})$")
                ax.set_ylabel(r"$y\ (\mu\mathrm{m})$")
                if row == 0:
                    ax.set_title(title)
                else:
                    ax.set_xlim((xc * 1e6) - half_window_um, (xc * 1e6) + half_window_um)
                    ax.set_ylim((yc * 1e6) - half_window_um, (yc * 1e6) + half_window_um)
                    ax.axhline(yc * 1e6, color="crimson", lw=0.8, ls="--")
                    ax.set_title("Zoom on a nucleus")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
                    "Intensity / background")

            # Profil horizontal à travers le noyau (ligne de pixels contenant yc)
            n = img.shape[0]
            centres = (np.arange(n) + 0.5) * self.L / n - self.L / 2
            row_idx = int(np.clip(np.floor((yc + self.L / 2) / (self.L / n)), 0, n - 1))
            sel = np.abs(centres - xc) <= half_window_um * 1e-6
            label = title.split("\n")[0]
            if col == 0:
                ax_p.plot(centres[sel] * 1e6, img[row_idx, sel], "k-", lw=1, label=label)
            else:
                ax_p.step(centres[sel] * 1e6, img[row_idx, sel], where="mid", label=label)
        ax_p.set_title("Horizontal profile through the nucleus (dashed line)")
        ax_p.set_xlabel(r"$x\ (\mu\mathrm{m})$")
        ax_p.set_ylabel("Intensity / background")
        ax_p.grid(True, ls=":", alpha=0.6)
        ax_p.legend()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    # ================= PART 5: Defocus and depth of field (Q7, Q8) =================
    def fresnelPropagate(self, field, dz, wavelength=None):
        """Q7 - Propagation libre sur dz par la méthode du spectre angulaire :
        E(z+dz) = TF^-1{ TF{E} * exp(i kz dz) }, kz = 2pi/lambda sqrt(1 - (lambda fx)^2 - (lambda fy)^2).
        Les ondes évanescentes (argument négatif) sont supprimées."""
        wl = self.lambda_ref if wavelength is None else wavelength
        f = np.fft.fftfreq(self.N, d=self.dx)
        FX, FY = np.meshgrid(f, f)

        arg = 1.0 - (wl * FX) ** 2 - (wl * FY) ** 2
        propagating = arg > 0
        kz = 2 * np.pi / wl * np.sqrt(np.maximum(arg, 0.0))
        H = np.exp(1j * kz * dz) * propagating

        return np.fft.ifft2(np.fft.fft2(field) * H)

    def getLayer2(self):
        """Deuxième couche : objet indépendant, mêmes paramètres, généré une seule
        fois (reproductible). Retourne (phase, intensité)."""
        if not hasattr(self, "_layer2"):
            self._layer2 = self.makeObject(
                n_cells=self.n_cells, cell_radius_mean=self.cell_radius_mean,
                cell_radius_sigma=self.cell_radius_sigma,
                nucleus_radius_frac=self.nucleus_radius_frac, delta_n=self.delta_n,
                lambda_ref=self.lambda_ref, L=self.L, N=self.N,
                nucleus_amplitude=self.nucleus_amplitude,
            )
        return self._layer2

    def simulate_two_layer(self, dz, NA):
        """
        Q8 - Objet à deux couches minces séparées de dz.
        La lumière traverse la couche 1 (t1), se propage sur dz, puis traverse la
        couche 2 (t2) : E = t2 * P_dz{t1} (les transmissions se multiplient).
        Retourne (image mise au point sur la couche 1, image mise au point sur la couche 2).
        Mise au point sur la couche 1 = rétro-propagation de E sur -dz avant la pupille.
        """
        t1 = self.get_complex_field()
        phase2, intensity2 = self.getLayer2()
        t2 = np.sqrt(intensity2) * np.exp(1j * phase2)

        E = t2 * self.fresnelPropagate(t1, dz)

        pupil = self.computePupil(NA, self.lambda_ref, self.L, self.N)
        focus_layer2 = self.imageWithPupil(pupil, E)
        focus_layer1 = self.imageWithPupil(pupil, self.fresnelPropagate(E, -dz))
        return focus_layer1, focus_layer2

    def _plotZoomGrid(self, columns, zooms, save_path, half_um=70):
        """Grille : ligne 1 = champ complet, lignes suivantes = zooms (centre, titre)."""
        fov_um = self.L * 1e6
        extent = [-fov_um / 2, fov_um / 2, -fov_um / 2, fov_um / 2]
        vmax = max(np.percentile(img, 99.9) for img, _ in columns)

        fig, axes = plt.subplots(1 + len(zooms), len(columns),
                                 figsize=(7.5 * len(columns), 5.8 * (1 + len(zooms))),
                                 constrained_layout=True, squeeze=False)
        for col, (img, title) in enumerate(columns):
            for row in range(1 + len(zooms)):
                ax = axes[row, col]
                im = ax.imshow(img, cmap="gray", extent=extent, origin="lower",
                               vmin=0, vmax=vmax)
                ax.set_xlabel(r"$x\ (\mu\mathrm{m})$")
                ax.set_ylabel(r"$y\ (\mu\mathrm{m})$")
                if row == 0:
                    ax.set_title(title)
                else:
                    (xc, yc), ztitle = zooms[row - 1]
                    ax.set_xlim(xc * 1e6 - half_um, xc * 1e6 + half_um)
                    ax.set_ylim(yc * 1e6 - half_um, yc * 1e6 + half_um)
                    ax.set_title(ztitle)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Intensity")
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def defocusComparison(self, dz=20e-6, NA=0.5, save_path="figures/part5_defocus.png"):
        """Q7 - Objet au point vs défocalisé de dz (brightfield)."""
        pupil = self.computePupil(NA, self.lambda_ref, self.L, self.N)
        field = self.get_complex_field()
        in_focus = self.imageWithPupil(pupil, field)
        defocused = self.imageWithPupil(pupil, self.fresnelPropagate(field, dz))
        dof_um = self.lambda_ref / NA**2 * 1e6
        print(f"=== DEFOCUS: dz = {dz*1e6:.0f} µm, NA = {NA}, "
              f"depth of field ~ lambda/NA^2 = {dof_um:.1f} µm ===")
        self._plotZoomGrid(
            [(in_focus, f"In focus (NA = {NA})"),
             (defocused, f"Defocused by dz = {dz*1e6:.0f} µm (DOF ≈ {dof_um:.1f} µm)")],
            [(self.nucleusCentre(), "Zoom on a cell")],
            save_path)

    def twoLayerComparison(self, dz=20e-6, NA=0.5, save_path="figures/part5_two_layers.png"):
        """Q8 - Objet à deux couches : mise au point sur la couche 1 puis sur la couche 2."""
        focus1, focus2 = self.simulate_two_layer(dz, NA)
        _, intensity2 = self.getLayer2()
        self._plotZoomGrid(
            [(focus1, f"Focus on layer 1 (z = 0), NA = {NA}"),
             (focus2, f"Focus on layer 2 (z = {dz*1e6:.0f} µm), NA = {NA}")],
            [(self.nucleusCentre(), "Zoom on a layer-1 cell"),
             (self.nucleusCentre(intensity_map=intensity2), "Zoom on a layer-2 cell")],
            save_path)


    # ================= PART 6 =================
    def assembleMicroscope(self, NA, dz=0.0, pixel_pitch=3.45e-6,
                           magnification=10.0, snr=50, field=None, seed=None):
        """Pipeline complet : propagation -> pupille -> détecteur avec bruit."""
        if field is None:
            field = self.get_complex_field()

        # 1. Défocus (si dz = 0, l'objet est exactement au plan focal)
        if dz != 0:
            field = self.fresnelPropagate(field, dz)

        # 2. Filtrage par la pupille de l'objectif
        pupil = self.computePupil(NA, self.lambda_ref, self.L, self.N)
        image = self.imageWithPupil(pupil, field)
        image = image / image.max()

        # 3. Caméra : intégration par pixel + bruit de Poisson
        return self.addCameraEffects(image, pixel_pitch, snr,
                                     magnification=magnification, seed=seed)



    def imageRealCells(self, image_path, NA, mode="amplitude", phi_max=1.0, **kwargs):
        raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f"Impossible de charger '{image_path}'.")
        img = cv2.resize(raw, (self.N, self.N)).astype(float) / 255.0

        if mode == "amplitude":      # image vue comme une transmission
            field = np.sqrt(img).astype(complex)
        else:                        # image vue comme une carte de phase
            field = np.exp(1j * phi_max * img)

        return self.assembleMicroscope(NA, field=field, **kwargs)

    



if __name__ == "__main__":

    np.random.seed(0)
    #  Cell sample
    sample = Microscopy(n_cells=8)

    # Maps
    sample.plot_obj()

    # ================= PART 2 : Pupil, imaging vs NA, USAF resolution =================
    sample.plotPupils()
    sample.compareNAs()
    sample.resolution()

    # ================= PART 3 : Contrast mechanisms (Q4, Q5) =================
    sample.contrastModes(NA=0.5)

    # ================= PART 4 : Detector sampling and noise (Q6) =================
    sample.cameraComparison(NA=0.5)

    # ================= PART 5 : Defocus and two-layer object (Q7, Q8) =================
    sample.defocusComparison(dz=20e-6, NA=0.5)
    sample.twoLayerComparison(dz=20e-6, NA=0.5)

    # ================= PARTIE 6 : Microscope complet (Q9, Q10, Q11) =================
    # Grille plus fine pour que la pupille de NA 0.25 tienne dans la grille de calcul
    hr = Microscopy(n_cells=5, L=0.4e-3, N=2048, NA=0.25)

    # Q10 : objectif 10X NA 0.25, caméra pixel 3.45 µm, SNR 60
    final = hr.assembleMicroscope(NA=0.25, dz=0, pixel_pitch=3.45e-6,
                                  magnification=10, snr=60, seed=0)

    fov_um = hr.L * 1e6
    extent_hr = [-fov_um / 2, fov_um / 2, -fov_um / 2, fov_um / 2]

    plt.figure(figsize=(6, 5), constrained_layout=True)
    plt.imshow(final, cmap="gray", extent=extent_hr, origin="lower",
               interpolation="nearest")
    plt.title("Image finale simulée (10X, NA = 0.25, pixel 3.45 µm, SNR = 60)")
    plt.xlabel(r"$x\ (\mu\mathrm{m})$")
    plt.ylabel(r"$y\ (\mu\mathrm{m})$")
    plt.colorbar(label="Intensité")
    plt.savefig("figures/part6_final_image.png", dpi=150)
    plt.show()

    # Q11 : image réelle de cellules (remplace par le nom du fichier fourni)
    real = hr.imageRealCells("Red Blood Cells DIC 8um.jpg", NA=0.25, pixel_pitch=3.45e-6,
                             magnification=10, snr=60, seed=0)

    plt.figure(figsize=(6, 5), constrained_layout=True)
    plt.imshow(real, cmap="gray", extent=extent_hr, origin="lower",
               interpolation="nearest")
    plt.title("Image réelle de cellules à travers le microscope virtuel")
    plt.xlabel(r"$x\ (\mu\mathrm{m})$")
    plt.ylabel(r"$y\ (\mu\mathrm{m})$")
    plt.colorbar(label="Intensité")
    plt.savefig("figures/part6_real_cells.png", dpi=150)
    plt.show()

