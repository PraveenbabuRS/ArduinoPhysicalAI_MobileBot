from setuptools import setup, find_packages

package_name = 'mobilebot_recorder'

setup(
    name=package_name,
    version='0.1.0',
    # find_packages picks up the vendored mobilebot_bridge.app_utils subpackage
    # automatically. Do not list packages by hand here or it will be dropped
    # from the install and the node will fail only at runtime.
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Praveenbabu R S',
    maintainer_email='praveenbabu4542@gmail.com',
    description='ROS 2 to MCAP data recording node.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'data_recorder = mobilebot_recorder.data_recorder:main',
        ],
    },
)
