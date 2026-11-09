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
        N=512,
        L=1.2e-3,
        NA=0.1,
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

        # 1. Grille spatiale
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
        N=512,
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
            intensity_map[mask_cell] = 0.85
            intensity_map[mask_nuc] = 0.60


        return phase_map, intensity_map

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



if __name__ == "__main__":
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
