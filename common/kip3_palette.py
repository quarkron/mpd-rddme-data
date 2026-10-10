"""The colour scheme for the Kip3 traffic figures: deep navy through blue and cyan to pale lime.

WHY IT IS SPLIT IN TWO
    The continuous ramp below carries the density fields - kymographs, overviews, profiles - and
    is a monotone lightness ramp, so it survives greyscale printing and reads the same to any
    colour vision. The three ARM colours that carry the series are drawn from the same family but
    chosen for separation rather than for the ramp: they differ in lightness (0.24 / 0.60 / 0.73)
    as well as hue, which is what keeps them apart under deuteranopia, the property jcp_style's
    own categorical set was checked for. Do not sample series colours off the ramp by eye; two
    neighbouring stops are indistinguishable as markers.

THE RENDERED SCENE IS ON WHITE
    The scene colours below are tuned for a white page, not for the dark ground the ramp starts
    at: the microtubule is a mid slate blue that stays readable once Lambertian shading takes its
    far flank down to about 40 % of base, and the motors are a deep teal so a 5 pt bead holds its
    edge against white. Putting the dark-ground values on white washes both out.
"""

# the ramp: near-black navy -> deep blue -> blue -> cyan -> pale yellow-green
STOPS = ['#050b23', '#0e2159', '#17509b', '#2093b4', '#41c5bd', '#a3dda0', '#e9f7bb']

# series colours: deep blue, teal, olive-lime
ARM = {'exc': '#123f7a', 'sw': '#2bb3a3', 'pt': '#9cbf3c', 'ref': '#5b6b7a'}

# the rendered scene, on white
GROUND = '#ffffff'          # panel background
TUBE = (0.50, 0.60, 0.74)   # microtubule base colour, before shading
TUBE_FILL = '#dfe7f0'       # the 3x3 body block in the cross-section
CAGE = '#93aac2'            # voxel lattice
MOTOR = (0.08, 0.53, 0.58)  # bound motor
INK = '#12202f'             # text
INK_DIM = '#5d7185'


def cmap(name='kip3_abyss'):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, STOPS)


def cmap_binary(name='kip3_ink'):
    """White to deep navy, for the binary occupancy strips."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, ['#ffffff', STOPS[1]])
