from setuptools import find_packages, setup

package_name = 'arm_pick_place'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='elephant',
    maintainer_email='elephant@todo.todo',
    description='Pick & place worker and commander for MyCobot 280 (MoveIt planning + HW/SIM execution).',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arm_worker = arm_pick_place.arm_worker:main',
            'commander = arm_pick_place.commander:main',
        ],
    },
)

