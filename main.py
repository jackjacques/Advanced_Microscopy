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
        N=1024,
        L=1.2e-3,
        NA=0.1,
        absorptive=False,
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
        self.absorptive = absorptive

        # --- AJOUTE CES 2 LIGNES ICI ---
        coords = np.linspace(-self.L / 2, self.L / 2, self.N)
        self.X, self.Y = np.meshgrid(coords, coords)

        # 2. Cartes d'objet
        self.phase_map, self.intensity_map = self.makeObject(
            n_cells=self.n_cells,
            cell_radius_mean=self.cell_radius_mean,
            cell_radius_sigma=self.cell_radius_sigma,
            nucleus_radius_frac=self.nucleus_radius_frac,
            delta_n=self.delta_n,
            lambda_ref=self.lambda_ref,
            L=self.L,
            N=self.N,
            absorptive=self.absorptive,
        )

        self.pupil = self.computePupil(self.NA, self.lambda_ref, self.L, self.N)

    @staticmethod
    def makeObject(
        n_cells=10,
        cell_radius_mean=50e-6,
        cell_radius_sigma=10e-6,
        nucleus_radius_frac=0.10,
        delta_n=0.02,
        lambda_ref=550e-9,
        L=1.2e-3,
        N=1024,
        absorptive=False,
    ):
        coords = np.linspace(-L / 2, L / 2, N)
        X, Y = np.meshgrid(coords, coords)

        phase_map = np.zeros((N, N))
        intensity_map = np.ones((N, N))

        k0 = 2 * np.pi / lambda_ref
        limit = L / 2 - cell_radius_mean

        for _ in range(n_cells):
            x0 = np.random.uniform(-limit, limit)
            y0 = np.random.uniform(-limit, limit)

            R_cell = max(
                10e-6, np.random.normal(cell_radius_mean, cell_radius_sigma)
            )
            R_nuc = R_cell * nucleus_radius_frac

            dist_sq = (X - x0) ** 2 + (Y - y0) ** 2

            mask_cell = dist_sq < R_cell**2
            d_cell = np.zeros_like(X)
            d_cell[mask_cell] = 2 * np.sqrt(R_cell**2 - dist_sq[mask_cell])

            mask_nuc = dist_sq < R_nuc**2
            d_nuc = np.zeros_like(X)
            d_nuc[mask_nuc] = 2 * np.sqrt(R_nuc**2 - dist_sq[mask_nuc])

            cell_phase = k0 * delta_n * d_cell + k0 * (delta_n * 0.5) * d_nuc
            phase_map = np.maximum(phase_map, cell_phase)
            if absorptive :
                intensity_map[mask_cell] = np.minimum(intensity_map[mask_cell], 0.85)
                intensity_map[mask_nuc] = np.minimum(intensity_map[mask_nuc], 0.60)


        return phase_map, intensity_map


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

    def save_figure_pupils(sample, NAs=[0.01, 0.1, 1.0]):
        fig, axes = plt.subplots(
            2, len(NAs), figsize=(14, 8), constrained_layout=True
        )
        dx = sample.L / sample.N
        fx_cpmm = np.fft.fftshift(np.fft.fftfreq(sample.N, d=dx)) * 1e-3
        extent = [fx_cpmm[0], fx_cpmm[-1], fx_cpmm[0], fx_cpmm[-1]]
        centre = sample.N // 2

        for col, na in enumerate(NAs):
            pupil = sample.computePupil(na, sample.lambda_ref, sample.L, sample.N)
            fc_cpmm = (na / sample.lambda_ref) * 1e-3

            # Ligne 1 : Pupille 2D
            im = axes[0, col].imshow(
                pupil, cmap="gray", extent=extent, origin="lower", vmin=0, vmax=1
            )
            axes[0, col].set_title(
                f"Pupille 2D (NA = {na})\n$f_c = {fc_cpmm:.1f}$ cycles/mm"
            )
            axes[0, col].set_xlabel(r"$f_x\ (\mathrm{cycles/mm})$")
            axes[0, col].set_ylabel(r"$f_y\ (\mathrm{cycles/mm})$")
            fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

            # Ligne 2 : Profil 1D
            axes[1, col].plot(
                fx_cpmm, pupil[centre, :], color="crimson", lw=2, label="Pupille"
            )
            axes[1, col].axvline(
                fc_cpmm,
                color="black",
                linestyle="--",
                alpha=0.7,
                label=f"+fc ({fc_cpmm:.0f})",
            )
            axes[1, col].axvline(-fc_cpmm, color="black", linestyle="--", alpha=0.7)
            axes[1, col].set_title(f"Coupe 1D ($f_y = 0$) - NA = {na}")
            axes[1, col].set_xlabel(r"$f_x\ (\mathrm{cycles/mm})$")
            axes[1, col].set_ylabel("Transmission")
            axes[1, col].set_ylim(-0.05, 1.1)
            axes[1, col].grid(True, linestyle=":", alpha=0.6)
            axes[1, col].legend(loc="upper right", fontsize=8)

        plt.savefig("figures/part2_pupil_and_cutoffs.png", dpi=150)
        plt.close()



    def plot_obj(self, figsize=(11, 4.5)):
        fov_um = self.L * 1e6
        extent = [-fov_um / 2, fov_um / 2, -fov_um / 2, fov_um / 2]

        fig, (ax_int, ax_phase) = plt.subplots(
            1, 2, figsize=figsize, constrained_layout=True
        )

        im_int = ax_int.imshow(
            self.intensity_map,
            cmap="gray",
            extent=extent,
            origin="lower",
            vmin=0,
            vmax=1,
        )
        ax_int.set_title(r"Intensité transmise $I(x, y)$")
        ax_int.set_xlabel(r"$x\ (\mu\mathrm{m})$")
        ax_int.set_ylabel(r"$y\ (\mu\mathrm{m})$")
        fig.colorbar(
            im_int, ax=ax_int, fraction=0.046, pad=0.04
        ).set_label("Transmission")

        im_phase = ax_phase.imshow(
            self.phase_map, cmap="viridis", extent=extent, origin="lower"
        )
        ax_phase.set_title(r"Carte de phase $\Delta\phi(x, y)$")
        ax_phase.set_xlabel(r"$x\ (\mu\mathrm{m})$")
        ax_phase.set_ylabel(r"$y\ (\mu\mathrm{m})$")
        fig.colorbar(
            im_phase, ax=ax_phase, fraction=0.046, pad=0.04
        ).set_label("Phase (rad)")

        plt.show()

    def resolution(
        self,
        image_path="usaf.png",
        NAs=[0.01, 0.1, 1.0],
    ):
        """Simule l'imagerie de la mire USAF et calcule les résolutions associées."""
        # 1. Chargement avec canal alpha éventuel (UNCHANGED)
        raw_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if raw_img is None:
            raise FileNotFoundError(
                f"Impossible de charger '{image_path}'. Vérifie le nom et le dossier."
            )

        # Gestion du canal alpha (transparence SVG -> PNG)
        if raw_img.ndim == 3 and raw_img.shape[2] == 4:
            # Fond blanc : là où alpha == 0, on met du blanc (255)
            alpha = raw_img[:, :, 3] / 255.0
            gray = cv2.cvtColor(raw_img[:, :, :3], cv2.COLOR_BGR2GRAY)
            img = (gray * alpha + 255 * (1 - alpha)).astype(np.uint8)
        elif raw_img.ndim == 3:
            img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        else:
            img = raw_img

        # 2. Redimensionnement et normalisation [0, 1]
        img_resized = cv2.resize(img, (self.N, self.N))
        target_amplitude = img_resized.astype(float) / 255.0

        # Si l'image est inversée (fond noir au lieu de blanc), on ré-inverse
        if np.mean(target_amplitude) < 0.5:
            target_amplitude = 1.0 - target_amplitude

        # 3. Champ d'onde incident et TF 2D
        E_in = target_amplitude.astype(complex)
        E_fft = np.fft.fftshift(np.fft.fft2(E_in))

        fov_um = self.L * 1e6
        extent = [-fov_um / 2, fov_um / 2, -fov_um / 2, fov_um / 2]

        fig, axes = plt.subplots(
            1, len(NAs) + 1, figsize=(16, 4.5), constrained_layout=True
        )

        # Affichage de la mire originale
        axes[0].imshow(
            target_amplitude,
            cmap="gray",
            extent=extent,
            origin="lower",
            vmin=0,
            vmax=1,
        )
        axes[0].set_title("Mire originale")
        axes[0].set_xlabel(r"$x\ (\mu\mathrm{m})$")
        axes[0].set_ylabel(r"$y\ (\mu\mathrm{m})$")

        print("=== RÉSULTATS DE RÉSOLUTION THÉORIQUE ===")
        # 4. Simulation pour chaque NA
        for ax, na in zip(axes[1:], NAs):
            pupil = self.computePupil(na, self.lambda_ref, self.L, self.N)
            E_filtered = E_fft * pupil
            E_out = np.fft.ifft2(np.fft.ifftshift(E_filtered))
            img_detected = np.abs(E_out) ** 2

            # Calculs des résolutions théoriques
            r_abbe = (self.lambda_ref / (2 * na)) * 1e6  # en µm
            r_rayleigh = (0.61 * self.lambda_ref / na) * 1e6  # en µm
            fc = na / self.lambda_ref  # Fréquence de coupure en m^-1

            print(f"NA = {na:4.2f} :")
            print(f"  - Coupure fc         : {fc*1e-3:.1f} cycles/mm")
            print(f"  - Limite Abbe (λ/2NA): {r_abbe:.2f} µm")
            print(f"  - Critère Rayleigh   : {r_rayleigh:.2f} µm")

            # Affichage normalisé pour éviter l'écran noir
            im = ax.imshow(
                img_detected,
                cmap="gray",
                extent=extent,
                origin="lower",
                vmin=np.min(img_detected),
                vmax=np.max(img_detected),
            )
            ax.set_title(f"NA = {na}\nRayleigh: {r_rayleigh:.2f} µm")
            ax.set_xlabel(r"$x\ (\mu\mathrm{m})$")
            ax.set_ylabel(r"$y\ (\mu\mathrm{m})$")

        print("=========================================")
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

    def contrastModes(self, NA=0.5, NA_in=0.05, phi_max_weak=0.3):
        """Q4 + Q5: compare brightfield, darkfield and phase contrast."""
        # Q4: brightfield vs darkfield on the real object
        bright = self.imageWithPupil(self.computePupil(NA, self.lambda_ref, self.L, self.N))
        dark = self.imageWithPupil(self.darkfieldPupil(NA_in, NA))

        # Q5: weak object -> linearization exp(i phi) ~ 1 + i phi
        # Our cells reach ~30 rad, far from "weak" (phi << 1):
        # we rescale the phase to phi_max_weak to stay in the validity domain.
        phi_weak = self.phase_map / self.phase_map.max() * phi_max_weak
        field_weak = np.sqrt(self.intensity_map) * (1 + 1j * phi_weak)
        bright_weak = self.imageWithPupil(
            self.computePupil(NA, self.lambda_ref, self.L, self.N), field_weak)
        phase_contrast = self.imageWithPupil(self.phaseContrastPupil(NA), field_weak)

        fov_um = self.L * 1e6
        extent = [-fov_um / 2, fov_um / 2, -fov_um / 2, fov_um / 2]
        images = [bright, dark, bright_weak, phase_contrast]
        titles = [f"Brightfield (NA = {NA})",
                  f"Darkfield ({NA_in} < NA <= {NA})",
                  f"Brightfield, weak object",
                  f"Phase contrast, weak object"]

        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), constrained_layout=True)
        for ax, img, title in zip(axes, images, titles):
            im = ax.imshow(img, cmap="gray", extent=extent, origin="lower")
            ax.set_title(title)
            ax.set_xlabel(r"$x\ (\mu\mathrm{m})$")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.savefig("part3_contrast_modes.png", dpi=150)
        plt.show()

    # ================= PART 4 : Échantillonnage du détecteur et bruit (Q6) =================

    def addCameraEffects(self, image, pixel_pitch, snr, magnification=1.0, seed=None):
        """Q6 - Simule une caméra : (1) intégration de la lumière par pixel,
        (2) bruit de photons (Poisson).

        pixel_pitch   : pas du pixel de la caméra [m]
        snr           : SNR au pic de l'image (nombre de photons au pic = snr^2)
        magnification : grossissement M. Dans le plan objet, un pixel
                        correspond à pixel_pitch / M.
        """
        # 1. Rééchantillonnage : taille du pixel ramenée dans le plan objet
        pitch_obj = pixel_pitch / magnification
        if pitch_obj < self.dx:
            print(f"Attention : pixel objet ({pitch_obj*1e6:.2f} µm) < dx de la "
                  f"grille ({self.dx*1e6:.2f} µm) : sur-échantillonnage, "
                  f"pas d'information en plus. Augmente N.")
        n_pix = int(round(self.L / pitch_obj))
        # INTER_AREA = moyenne sur la surface couverte par chaque pixel,
        # ce qui reproduit l'intégration de la lumière par un vrai pixel
        resampled = cv2.resize(image.astype(np.float64), (n_pix, n_pix),
                               interpolation=cv2.INTER_AREA)
        resampled = np.clip(resampled, 0, None)

        # 2. Bruit de Poisson : SNR = sqrt(N)  =>  N_pic = SNR^2 photons
        peak_photons = snr**2
        scaled = resampled / max(resampled.max(), 1e-12) * peak_photons
        rng = np.random.default_rng(seed)
        noisy = rng.poisson(scaled)

        # 3. Retour à une échelle relative (pic ~ 1) pour l'affichage
        return noisy / peak_photons

    def cameraComparison(self, NA=0.5, cameras=((1e-6, 30), (3.5e-6, 60)),
                         magnification=1.0, half_window_um=30):
        """Q6 - Même objet, deux caméras : (pixel 1 µm, SNR 30) vs (pixel 3.5 µm, SNR 60).
        Ligne du haut : champ complet. Ligne du bas : zoom sur un noyau."""
        ideal = self.applyPupilImaging(NA)
        ideal = ideal / ideal.max()

        images = [ideal]
        titles = [f"Image idéale\n(dx = {self.dx*1e6:.2f} µm, sans bruit)"]

        nyquist_um = self.lambda_ref / (4 * NA) * 1e6
        nuc_diam_um = 2 * self.nucleus_radius_frac * self.cell_radius_mean * 1e6
        print(f"=== CAMÉRAS (NA = {NA}, M = {magnification}) ===")
        print(f"Nyquist pour la NA : pixel objet <= lambda/(4NA) = {nyquist_um:.2f} µm")
        for pitch, snr in cameras:
            p_obj = pitch / magnification * 1e6
            print(f"Pixel {pitch*1e6:.1f} µm, SNR {snr} : pixel objet = {p_obj:.2f} µm, "
                  f"{snr**2} photons au pic, noyau ~ {nuc_diam_um/p_obj:.1f} pixels de large")
            images.append(self.addCameraEffects(ideal, pitch, snr, magnification, seed=0))
            titles.append(f"Pixel {pitch*1e6:.1f} µm, SNR = {snr}")

        fov_um = self.L * 1e6
        extent = [-fov_um / 2, fov_um / 2, -fov_um / 2, fov_um / 2]

        # Zoom centré sur un noyau (premier pixel de transmission minimale)
        iy, ix = np.unravel_index(np.argmin(self.intensity_map), self.intensity_map.shape)
        xc, yc = self.X[iy, ix] * 1e6, self.Y[iy, ix] * 1e6

        fig, axes = plt.subplots(2, len(images), figsize=(5 * len(images), 9),
                                 constrained_layout=True)
        for col, (img, title) in enumerate(zip(images, titles)):
            for row in range(2):
                ax = axes[row, col]
                ax.imshow(img, cmap="gray", extent=extent, origin="lower",
                          interpolation="nearest", vmin=0.3, vmax=1.1)
                ax.set_xlabel(r"$x\ (\mu\mathrm{m})$")
                if row == 0:
                    ax.set_title(title)
                else:
                    ax.set_xlim(xc - half_window_um, xc + half_window_um)
                    ax.set_ylim(yc - half_window_um, yc + half_window_um)
        plt.savefig("part4_camera.png", dpi=150)
        plt.show()
    # ================= PART 5  =================
    def fresnelPropagate(self, field, dz, wavelength=None):
        
        wl = self.lambda_ref if wavelength is None else wavelength
        f = np.fft.fftfreq(self.N, d=self.dx)
        FX, FY = np.meshgrid(f, f)

        arg = 1.0 - (wl * FX) ** 2 - (wl * FY) ** 2
        propagating = arg > 0                       # ondes évanescentes (arg < 0) : supprimées
        kz = 2 * np.pi / wl * np.sqrt(np.maximum(arg, 0.0))
        H = np.exp(1j * kz * dz) * propagating

        return np.fft.ifft2(np.fft.fft2(field) * H)

    def simulate_two_layer(self, dz, NA):
        """
        Simule un objet à deux couches séparées par dz et passe le tout 
        à travers le système d'imagerie du microscope (pupille et NA).
        """
        # 1. Champ de la couche 1 (dans le plan focal z = 0)
        E1 = self.get_complex_field()
        
        # 2. Génération d'une deuxième couche (deuxième objet indépendant) 
        # et propagation de Fresnel sur la distance dz
        phase2, intensity2 = self.makeObject() # Adaptez selon votre méthode de création d'objet
        E2_initial = np.sqrt(intensity2) * np.exp(1j * phase2)
        E2_propagated = self.fresnelPropagate(E2_initial, dz=dz)
        
        # 3. Superposition des deux champs au niveau de l'objet
        E_total = E1 + E2_propagated
        
        # 4. Passage à travers le système optique du microscope (Pupille / NA)
        pupil = self.computePupil(NA=NA, lambda_ref=self.lambda_ref, L=self.L, N=self.N)
        image_two_layers = self.imageWithPupil(pupil, field=E_total)
        
        return image_two_layers
    

    



if __name__ == "__main__":

    np.random.seed(0)
    #  Cell sample
    sample = Microscopy(n_cells=8)

    # Maps
    sample.plot_obj()

    # USAF resolutions
    sample.resolution("usaf.png")

    NAs = [0.01, 0.1, 1.0]
    fov_um = sample.L * 1e6
    extent = [-fov_um / 2, fov_um / 2, -fov_um / 2, fov_um / 2]

    fig, axes = plt.subplots(
        1, len(NAs), figsize=(15, 4.5), constrained_layout=True
    )

    for ax, na in zip(axes, NAs):
        image = sample.applyPupilImaging(na)
        im = ax.imshow(image, cmap="gray", extent=extent, origin="lower")
        ax.set_title(f"Cellules détectées (NA = {na})")
        ax.set_xlabel(r"$x\ (\mu\mathrm{m})$")
        ax.set_ylabel(r"$y\ (\mu\mathrm{m})$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.show()
    
    sample.contrastModes(NA=0.5)

    # ================= PARTIE 4 : Échantillonnage et Bruit (Q6) =================
    sample.cameraComparison(NA=0.5)

    # ================= PARTIE 5 : Défocalisation et Profondeur de Champ (Q7) =================
    dz = 20e-6  # Défocalisation de 20 µm
    field_in = sample.get_complex_field()
    
    # Propagation de Fresnel en amont du système d'imagerie
    field_defocused = sample.fresnelPropagate(field_in, dz=dz)
    
    # Imagerie du champ défocalisé avec une NA de 0.5
    pupil = sample.computePupil(NA=0.5, lambda_ref=sample.lambda_ref, L=sample.L, N=sample.N)
    image_defocused = sample.imageWithPupil(pupil, field=field_defocused)

    image_two_layers = sample.simulate_two_layer(dz=dz, NA=0.5)
    print("Image générée avec succès, dimensions :", image_two_layers.shape)

    # Affichage de l'image défocalisée
    plt.figure(figsize=(6, 5), constrained_layout=True)
    plt.imshow(image_defocused, cmap="gray", extent=extent, origin="lower")
    plt.title(f"Image défocalisée (dz = {dz*1e6:.0f} µm, NA = 0.5)")
    plt.xlabel(r"$x\ (\mu\mathrm{m})$")
    plt.ylabel(r"$y\ (\mu\mathrm{m})$")
    plt.colorbar(label="Intensité")
    plt.show()   

    # Affichage de l'objet à deux couches
    # affichage
    plt.figure(figsize=(6, 5), constrained_layout=True)
    plt.imshow(image_two_layers, cmap="gray", extent=extent, origin="lower")
    plt.title(f"Objet à deux couches séparées de {dz*1e6:.0f} µm (NA = 0.5)")
    plt.xlabel(r"$x\ (\mu\mathrm{m})$")
    plt.ylabel(r"$y\ (\mu\mathrm{m})$")
    plt.colorbar(label="Intensité")
    plt.show()

