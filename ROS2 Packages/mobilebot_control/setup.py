from setuptools import setup

package_name = 'mobilebot_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Praveenbabu R S',
    maintainer_email='praveenbabu4542@gmail.com',
    description='Teleoperation translation and safety arbitration.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'teleop_translator = mobilebot_control.teleop_translator:main',
            'safety_arbiter = mobilebot_control.safety_arbiter:main',
        ],
    },
)
