from setuptools import setup, find_namespace_packages, find_packages


setup(
    name = "kutu",
    version = "0.1",
    author = "Ilker Bozcan",
    author_email = "ilker@eng.au.dk",
    description = ("Machine Learning toolbox for my Ph.D."),
<<<<<<< e2d71508555f5db8f225023deb010c41c2864bdd
    packages=['kutu']
=======
    packages=['kutu'], install_requires=['numpy', 'wget', 'cv2']
>>>>>>> image classification dataset
)
