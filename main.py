import matplotlib.pyplot as plt
import numpy as np
import cv2


# Q10 - Composants retenus pour le microscope (références à vérifier sur thorlabs.com)
OBJECTIVE = dict(
    name="10X / NA 0.25 objective (e.g. Olympus RMS10X, sold by Thorlabs)",
    magnification=10.0,
    NA=0.25,
)
CAMERA = dict(
    name="Thorlabs Zelux CS165MU (Sony IMX273 sensor)",
    pixel_pitch=3.45e-6,
    n_cols=1440,
    n_rows=1080,
    snr=60,
)


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
        """Q2 - Pupilles binaires pour chaque NA."""
        fx_cpmm = np.fft.fftshift(np.fft.fftfreq(self.N, d=self.dx)) * 1e-3
        extent = [fx_cpmm[0], fx_cpmm[-1], fx_cpmm[0], fx_cpmm[-1]]
        fig, axes = plt.subplots(1, len(NAs), figsize=(5 * len(NAs), 4.5), constrained_layout=True)
        for ax, na in zip(axes, NAs):
            pupil = self.computePupil(na, self.lambda_ref, self.L, self.N)
            ax.imshow(pupil, cmap="gray", extent=extent, origin="lower")
            ax.set_title(f"Pupil, NA = {na}")
            ax.set_xlabel(r"$f_x$ (cycles/mm)")
            ax.set_ylabel(r"$f_y$ (cycles/mm)")
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def compareNAs(self, NAs=(0.01, 0.1, 1.0), save_path="figures/part2_cells_vs_NA.png"):
        """Q3 - Image de l'objet pour plusieurs NA."""
        fov_um = self.L * 1e6
        extent = [-fov_um / 2, fov_um / 2, -fov_um / 2, fov_um / 2]
        fig, axes = plt.subplots(1, len(NAs), figsize=(5 * len(NAs), 4.5), constrained_layout=True)
        for ax, na in zip(axes, NAs):
            ax.imshow(self.applyPupilImaging(na), cmap="gray", extent=extent, origin="lower")
            ax.set_title(f"NA = {na}")
            ax.set_xlabel(r"$x\ (\mu\mathrm{m})$")
            ax.set_ylabel(r"$y\ (\mu\mathrm{m})$")
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
        Q3.3 - Mire USAF 1951 à l'échelle réelle, imagée pour chaque NA.
        Pour chaque NA, on affiche le plus petit élément encore résolu
        (contraste entre barres et interstices >= threshold).
        Ligne 1 : mire entière. Ligne 2 : zoom sur les petits groupes (6 à 9).
        """
        target, elements = self.makeUSAFTarget(groups)
        E_fft = np.fft.fftshift(np.fft.fft2(target.astype(complex)))
        fov_um = self.L * 1e6
        extent = [-fov_um / 2, fov_um / 2, -fov_um / 2, fov_um / 2]

        # Zone des petits groupes pour le zoom
        small = [el for el in elements if el["group"] >= 6]
        zx = (min(el["x0"] for el in small), max(el["x0"] + 5 * el["w"] for el in small))
        zy = (min(el["y0"] for el in small), max(el["y0"] + 5 * el["w"] for el in small))

        images = [(target, "USAF 1951 target")]
        print("=== USAF 1951 RESOLUTION ===")
        for na in NAs:
            pupil = self.computePupil(na, self.lambda_ref, self.L, self.N)
            img = np.abs(np.fft.ifft2(np.fft.ifftshift(E_fft * pupil))) ** 2
            C = np.array([self.elementContrast(img, el) for el in elements])
            failed = np.flatnonzero(C < threshold)
            last = (failed[0] if failed.size else len(elements)) - 1
            el = elements[max(last, 0)]
            print(f"NA = {na}: smallest resolved element = group {el['group']} element "
                  f"{el['element']} ({el['lp_mm']:.0f} lp/mm, line width {el['w']*1e6:.2f} µm)")
            images.append((img, f"NA = {na}"))

        fig, axes = plt.subplots(2, len(images), figsize=(4.5 * len(images), 7),
                                 constrained_layout=True)
        for col, (img, title) in enumerate(images):
            for row in range(2):
                ax = axes[row, col]
                ax.imshow(img, cmap="gray", extent=extent, origin="lower")
                ax.set_xlabel(r"$x\ (\mu\mathrm{m})$")
                if row == 0:
                    ax.set_title(title)
                else:
                    ax.set_xlim(zx[0] * 1e6 - 5, zx[1] * 1e6 + 5)
                    ax.set_ylim(zy[0] * 1e6 - 5, zy[1] * 1e6 + 5)
                    ax.set_title(f"Zoom on groups 6-9")
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
        Le contraste de phase suppose un objet de phase faible : on réduit la phase
        de l'objet à phi_max_weak (nos cellules atteignent ~30 rad).
        """
        bright = self.imageWithPupil(self.computePupil(NA, self.lambda_ref, self.L, self.N))
        dark = self.imageWithPupil(self.darkfieldPupil(NA_in, NA))
        phi_weak = self.phase_map / self.phase_map.max() * phi_max_weak
        field_weak = np.sqrt(self.intensity_map) * np.exp(1j * phi_weak)
        phc = self.imageWithPupil(self.phaseContrastPupil(NA), field_weak)
        fov_um = self.L * 1e6
        extent = [-fov_um / 2, fov_um / 2, -fov_um / 2, fov_um / 2]
        panels = [(bright, f"Brightfield (NA = {NA})"),
                  (dark, f"Darkfield ({NA_in} < NA < {NA})"),
                  (phc, f"Phase contrast (phase reduced to {phi_max_weak} rad)")]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
        for ax, (img, title) in zip(axes, panels):
            ax.imshow(img, cmap="gray", extent=extent, origin="lower")
            ax.set_title(title)
            ax.set_xlabel(r"$x\ (\mu\mathrm{m})$")
            ax.set_ylabel(r"$y\ (\mu\mathrm{m})$")
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

    def addCameraEffects(self, image, pixel_pitch, snr, magnification=1.0, seed=None,
                         reference="background"):
        """Q6 - Simule une caméra : (1) intégration de la lumière par pixel,
        (2) bruit de photons (Poisson).

        pixel_pitch   : pas du pixel de la caméra [m]
        snr           : SNR sur le fond de l'image : le fond reçoit snr^2 photons
                        (Poisson : SNR = N / sqrt(N) = sqrt(N))
        magnification : grossissement M. Dans le plan objet, un pixel
                        correspond à pixel_pitch / M.

        reference     : "background" (médiane, défaut) ou "peak" (99.9e percentile,
                        pour le fond noir du darkfield où la médiane vaut ~0).

        Retourne l'image bruitée normalisée par la référence (référence ~ 1).
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
        if reference == "peak":
            background = max(np.percentile(resampled, 99.9), 1e-12)
        else:
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
        Ligne 1 : champ complet. Ligne 2 : zoom sur un noyau."""
        ideal = self.applyPupilImaging(NA)
        ideal = ideal / np.median(ideal)  # fond = 1
        images = [(ideal, "Ideal image (no camera)")]
        for pitch, snr in cameras:
            images.append((self.addCameraEffects(ideal, pitch, snr, magnification, seed=0),
                           f"Pixel {pitch*1e6:.1f} µm, SNR = {snr}"))

        xc, yc = (c * 1e6 for c in self.nucleusCentre())
        fov_um = self.L * 1e6
        extent = [-fov_um / 2, fov_um / 2, -fov_um / 2, fov_um / 2]
        fig, axes = plt.subplots(2, len(images), figsize=(5 * len(images), 9),
                                 constrained_layout=True)
        for col, (img, title) in enumerate(images):
            for row in range(2):
                ax = axes[row, col]
                ax.imshow(img, cmap="gray", extent=extent, origin="lower",
                          interpolation="nearest", vmin=0.75, vmax=1.25)
                ax.set_xlabel(r"$x\ (\mu\mathrm{m})$")
                ax.set_ylabel(r"$y\ (\mu\mathrm{m})$")
                if row == 0:
                    ax.set_title(title)
                else:
                    ax.set_xlim(xc - half_window_um, xc + half_window_um)
                    ax.set_ylim(yc - half_window_um, yc + half_window_um)
                    ax.set_title("Zoom on a nucleus")
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

    def _plotSideBySide(self, panels, save_path):
        """Images côte à côte, même échelle de gris."""
        fov_um = self.L * 1e6
        extent = [-fov_um / 2, fov_um / 2, -fov_um / 2, fov_um / 2]
        vmax = max(np.percentile(img, 99.9) for img, _ in panels)
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5),
                                 constrained_layout=True)
        for ax, (img, title) in zip(axes, panels):
            ax.imshow(img, cmap="gray", extent=extent, origin="lower", vmin=0, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel(r"$x\ (\mu\mathrm{m})$")
            ax.set_ylabel(r"$y\ (\mu\mathrm{m})$")
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def defocusComparison(self, dz=20e-6, NA=0.5, save_path="figures/part5_defocus.png"):
        """Q7 - Objet au point vs défocalisé de dz."""
        pupil = self.computePupil(NA, self.lambda_ref, self.L, self.N)
        field = self.get_complex_field()
        in_focus = self.imageWithPupil(pupil, field)
        defocused = self.imageWithPupil(pupil, self.fresnelPropagate(field, dz))
        self._plotSideBySide([(in_focus, f"In focus (NA = {NA})"),
                              (defocused, f"Defocused by {dz*1e6:.0f} µm")], save_path)

    def twoLayerComparison(self, dz=20e-6, NA=0.5, save_path="figures/part5_two_layers.png"):
        """Q8 - Objet à deux couches : mise au point sur la couche 1 puis sur la couche 2."""
        focus1, focus2 = self.simulate_two_layer(dz, NA)
        self._plotSideBySide([(focus1, "Focus on layer 1"),
                              (focus2, f"Focus on layer 2 ({dz*1e6:.0f} µm further)")],
                             save_path)


    # ================= PART 6: Complete microscope (Q9, Q10, Q11) =================
    def modePupil(self, NA, mode="brightfield", NA_in=0.05):
        """Pupille selon le mode d'imagerie (Parties 2 et 3)."""
        if mode == "brightfield":
            return self.computePupil(NA, self.lambda_ref, self.L, self.N)
        if mode == "darkfield":
            return self.darkfieldPupil(NA_in, NA)
        if mode == "phase":
            return self.phaseContrastPupil(NA)
        raise ValueError(f"Unknown mode '{mode}' (brightfield, darkfield, phase)")

    def assembleMicroscope(self, NA, dz=0.0, pixel_pitch=3.45e-6, magnification=10.0,
                           snr=60, field=None, seed=None, mode="brightfield", NA_in=0.05,
                           sensor_shape=None, centre=(0.0, 0.0)):
        """
        Q9 - Pipeline complet :
        (1) propagation (défocus dz, 0 = objet au point),
        (2) filtrage par la pupille de l'objectif (brightfield / darkfield / phase),
        (3) caméra : intégration par pixel (pitch / M) + bruit de Poisson,
        (4) recadrage sur le champ du capteur (sensor_shape = (lignes, colonnes))
            centré sur `centre` (m, plan objet).
        Retourne (image, extent en µm).
        """
        if field is None:
            field = self.get_complex_field()
        if dz != 0:
            field = self.fresnelPropagate(field, dz)

        image = self.imageWithPupil(self.modePupil(NA, mode, NA_in), field)

        reference = "peak" if mode == "darkfield" else "background"
        cam = self.addCameraEffects(image, pixel_pitch, snr, magnification, seed=seed,
                                    reference=reference)

        # Recadrage sur le capteur
        n = cam.shape[0]
        p = self.L / n  # pixel effectif dans le plan objet
        r0, c0 = 0, 0
        rows, cols = n, n
        if sensor_shape is not None:
            rows, cols = min(sensor_shape[0], n), min(sensor_shape[1], n)
            ic = int(round((centre[1] + self.L / 2) / p))
            jc = int(round((centre[0] + self.L / 2) / p))
            r0 = int(np.clip(ic - rows // 2, 0, n - rows))
            c0 = int(np.clip(jc - cols // 2, 0, n - cols))
            cam = cam[r0:r0 + rows, c0:c0 + cols]
        x0 = (-self.L / 2 + c0 * p) * 1e6
        y0 = (-self.L / 2 + r0 * p) * 1e6
        extent = [x0, x0 + cols * p * 1e6, y0, y0 + rows * p * 1e6]
        return cam, extent

    def finalMicroscope(self, objective=OBJECTIVE, camera=CAMERA,
                        save_path="figures/part6_final_image.png"):
        """Q10 - Image finale avec les composants choisis (brightfield)."""
        M, NA = objective["magnification"], objective["NA"]
        cells = self.phase_map > 0
        centre = (self.X[cells].mean(), self.Y[cells].mean())
        img, extent = self.assembleMicroscope(
            NA, pixel_pitch=camera["pixel_pitch"], magnification=M, snr=camera["snr"], seed=0,
            sensor_shape=(camera["n_rows"], camera["n_cols"]), centre=centre)
        fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
        ax.imshow(img, cmap="gray", extent=extent, origin="lower", interpolation="nearest")
        ax.set_title(f"{M:.0f}X / NA {NA}, pixel {camera['pixel_pitch']*1e6:.2f} µm, "
                     f"SNR = {camera['snr']}")
        ax.set_xlabel(r"$x\ (\mu\mathrm{m})$")
        ax.set_ylabel(r"$y\ (\mu\mathrm{m})$")
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def realCellsComparison(self, image_path="Red Blood Cells DIC 8um.jpg",
                            cell_diameter=8e-6, cell_diameter_px=70,
                            objective=OBJECTIVE, camera=CAMERA,
                            save_path="figures/part6_real_cells.png"):
        """
        Q11 - Image réelle de globules rouges à travers le microscope.
        Échelle : un globule rouge (8 µm) fait ~70 pixels sur la photo.
        La photo est utilisée comme transmission en amplitude.
        """
        raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f"Cannot load '{image_path}'.")
        img = np.flipud(raw.astype(float))  # ligne 0 = bas, comme la grille
        bg = np.median(img)

        px = cell_diameter / cell_diameter_px
        N = int(2 ** np.ceil(np.log2(max(img.shape))))
        grid = Microscopy(n_cells=0, L=N * px, N=N, lambda_ref=self.lambda_ref)
        canvas = np.full((N, N), bg)
        r0, c0 = (N - img.shape[0]) // 2, (N - img.shape[1]) // 2
        canvas[r0:r0 + img.shape[0], c0:c0 + img.shape[1]] = img

        M, NA = objective["magnification"], objective["NA"]
        sim, extent = grid.assembleMicroscope(
            NA, field=np.sqrt(canvas / bg).astype(complex), pixel_pitch=camera["pixel_pitch"],
            magnification=M, snr=camera["snr"], seed=0,
            sensor_shape=(camera["n_rows"], camera["n_cols"]))

        w, h = img.shape[1] * px * 1e6, img.shape[0] * px * 1e6
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for ax, (im_, ext, title) in zip(axes, [
                (canvas, [-N * px * 5e5, N * px * 5e5] * 2, "Original image"),
                (sim, extent, f"Through our microscope ({M:.0f}X / NA {NA})")]):
            ax.imshow(im_, cmap="gray", extent=ext, origin="lower", interpolation="nearest")
            ax.set_xlim(-w / 2, w / 2)
            ax.set_ylim(-h / 2, h / 2)
            ax.set_title(title)
            ax.set_xlabel(r"$x\ (\mu\mathrm{m})$")
            ax.set_ylabel(r"$y\ (\mu\mathrm{m})$")
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


if __name__ == "__main__":

    np.random.seed(0)
    #  Cell sample
    sample = Microscopy(n_cells=10)

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

    # ================= PART 6 : Complete microscope (Q9, Q10, Q11) =================
    sample.finalMicroscope()
    sample.realCellsComparison("Red Blood Cells DIC 8um.jpg")
