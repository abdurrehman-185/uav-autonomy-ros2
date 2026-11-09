#!/usr/bin/env python3
"""Generates Gazebo worlds containing the X3-style quadrotor with a chosen sensor suite.

Flight parameters (rotor constants, controller gains, inertia) follow the X3 quadcopter in
Gazebo's multicopter_velocity_control.sdf example.
"""

ROTORS = [  # name, x, y, turning direction, controller direction
    ('rotor_0', 0.13, -0.22, 'ccw', 1),
    ('rotor_1', -0.13, 0.2, 'ccw', 1),
    ('rotor_2', 0.13, 0.22, 'cw', -1),
    ('rotor_3', -0.13, -0.2, 'cw', -1),
]


def noise(std, bias_mean=0.0, bias_std=0.0):
    return (f'<noise type="gaussian"><mean>0</mean><stddev>{std}</stddev>'
            f'<bias_mean>{bias_mean}</bias_mean><bias_stddev>{bias_std}</bias_stddev></noise>')


def imu_sensor(rate, accel_std, accel_bias, gyro_std, gyro_bias):
    axes = lambda std, bias: ''.join(f'<{a}>{noise(std, bias, bias * 0.2)}</{a}>' for a in 'xyz')
    return f'''
        <sensor name="imu" type="imu">
          <always_on>1</always_on><update_rate>{rate}</update_rate><topic>imu</topic>
          <imu>
            <angular_velocity>{axes(gyro_std, gyro_bias)}</angular_velocity>
            <linear_acceleration>{axes(accel_std, accel_bias)}</linear_acceleration>
          </imu>
        </sensor>'''


def gps_sensor(rate, horizontal_std, vertical_std):
    """horizontal_std and vertical_std in meters. Gazebo applies horizontal NavSat noise to
    latitude/longitude in degrees, so it is converted here (worlds sit near the equator,
    where a degree of latitude and of longitude are both about 111.32 km)."""
    horizontal_std = horizontal_std / 111319.5
    return f'''
        <sensor name="navsat" type="navsat">
          <always_on>1</always_on><update_rate>{rate}</update_rate><topic>navsat</topic>
          <navsat>
            <position_sensing>
              <horizontal><noise type="gaussian"><mean>0</mean><stddev>{horizontal_std}</stddev></noise></horizontal>
              <vertical><noise type="gaussian"><mean>0</mean><stddev>{vertical_std}</stddev></noise></vertical>
            </position_sensing>
          </navsat>
        </sensor>'''


def lidar_sensor(max_range=30.0):
    return f'''
        <sensor name="lidar" type="gpu_lidar">
          <pose>0 0 0.12 0 0 0</pose><always_on>1</always_on><update_rate>10</update_rate><topic>scan</topic>
          <lidar>
            <scan><horizontal><samples>360</samples><resolution>1</resolution><min_angle>-3.14159</min_angle><max_angle>3.12414</max_angle></horizontal></scan>
            <range><min>0.45</min><max>{max_range}</max><resolution>0.01</resolution></range>
            <noise><type>gaussian</type><mean>0</mean><stddev>0.015</stddev></noise>
          </lidar>
        </sensor>'''


def range_sensor():
    return '''
        <sensor name="range" type="gpu_lidar">
          <pose>0 0 -0.06 0 1.5708 0</pose><always_on>1</always_on><update_rate>20</update_rate><topic>range</topic>
          <lidar>
            <scan><horizontal><samples>1</samples><resolution>1</resolution><min_angle>0</min_angle><max_angle>0</max_angle></horizontal></scan>
            <range><min>0.02</min><max>40</max><resolution>0.01</resolution></range>
            <noise><type>gaussian</type><mean>0</mean><stddev>0.01</stddev></noise>
          </lidar>
        </sensor>'''


def camera(name, pose, fov, far, rate=15):
    return f'''
        <sensor name="{name}" type="camera">
          <pose>{pose}</pose><always_on>1</always_on><update_rate>{rate}</update_rate><topic>{name}</topic>
          <camera>
            <horizontal_fov>{fov}</horizontal_fov>
            <image><width>640</width><height>360</height><format>R8G8B8</format></image>
            <clip><near>0.1</near><far>{far}</far></clip>
          </camera>
        </sensor>'''


def drone(sensors_xml, pose='0 0 0.1 0 0 0', color='0.85 0.15 0.1 1'):
    inertia_rotor = ('<inertial><mass>0.005</mass><inertia><ixx>9.75e-07</ixx><ixy>0</ixy><ixz>0</ixz>'
                     '<iyy>0.000273104</iyy><iyz>0</iyz><izz>0.000274004</izz></inertia></inertial>')
    rotor_links = ''.join(f'''
      <link name="{n}"><pose>{x} {y} 0.023 0 0 0</pose>{inertia_rotor}
        <visual name="v"><geometry><cylinder><radius>0.1</radius><length>0.005</length></cylinder></geometry>
          <material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.1 0.1 0.1 1</diffuse></material></visual></link>
      <joint name="{n}_joint" type="revolute"><parent>base_link</parent><child>{n}</child>
        <axis><xyz>0 0 1</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis></joint>''' for n, x, y, _, _ in ROTORS)
    motors = ''.join(f'''
      <plugin filename="gz-sim-multicopter-motor-model-system" name="gz::sim::systems::MulticopterMotorModel">
        <robotNamespace>X3</robotNamespace><jointName>{n}_joint</jointName><linkName>{n}</linkName>
        <turningDirection>{turn}</turningDirection><timeConstantUp>0.0125</timeConstantUp><timeConstantDown>0.025</timeConstantDown>
        <maxRotVelocity>800.0</maxRotVelocity><motorConstant>8.54858e-06</motorConstant><momentConstant>0.016</momentConstant>
        <commandSubTopic>gazebo/command/motor_speed</commandSubTopic><actuator_number>{i}</actuator_number>
        <rotorDragCoefficient>8.06428e-05</rotorDragCoefficient><rollingMomentCoefficient>1e-06</rollingMomentCoefficient>
        <motorSpeedPubTopic>motor_speed/{i}</motorSpeedPubTopic><rotorVelocitySlowdownSim>10</rotorVelocitySlowdownSim>
        <motorType>velocity</motorType>
      </plugin>''' for i, (n, _, _, turn, _) in enumerate(ROTORS))
    rotor_cfg = ''.join(f'<rotor><jointName>{n}_joint</jointName><forceConstant>8.54858e-06</forceConstant>'
                        f'<momentConstant>0.016</momentConstant><direction>{d}</direction></rotor>' for n, _, _, _, d in ROTORS)
    return f'''
    <model name="x3">
      <pose>{pose}</pose>
      <link name="base_link">
        <inertial><mass>1.5</mass>
          <inertia><ixx>0.0347563</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0.07</iyy><iyz>0</iyz><izz>0.0977</izz></inertia></inertial>
        <collision name="body_c"><geometry><box><size>0.47 0.47 0.11</size></box></geometry></collision>
        <visual name="body_v"><geometry><box><size>0.3 0.12 0.08</size></box></geometry>
          <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>
        <visual name="arm1_v"><pose>0 0 0 0 0 1.03</pose><geometry><box><size>0.5 0.02 0.02</size></box></geometry>
          <material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.1 0.1 0.1 1</diffuse></material></visual>
        <visual name="arm2_v"><pose>0 0 0 0 0 -1.03</pose><geometry><box><size>0.5 0.02 0.02</size></box></geometry>
          <material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.1 0.1 0.1 1</diffuse></material></visual>
        <visual name="top_v"><pose>0 0 0.08 0 0 0</pose><geometry><cylinder><radius>0.04</radius><length>0.05</length></cylinder></geometry>
          <material><ambient>0.1 0.1 0.8 1</ambient><diffuse>0.1 0.1 0.8 1</diffuse></material></visual>
{sensors_xml}
      </link>
{rotor_links}
{motors}
      <plugin filename="gz-sim-multicopter-control-system" name="gz::sim::systems::MulticopterVelocityControl">
        <robotNamespace>X3</robotNamespace><commandSubTopic>gazebo/command/twist</commandSubTopic>
        <enableSubTopic>enable</enableSubTopic><comLinkName>base_link</comLinkName>
        <velocityGain>2.7 2.7 2.7</velocityGain><attitudeGain>2 3 0.15</attitudeGain>
        <angularRateGain>0.4 0.52 0.18</angularRateGain><maximumLinearAcceleration>2 2 2</maximumLinearAcceleration>
        <rotorConfiguration>{rotor_cfg}</rotorConfiguration>
      </plugin>
      <!-- Ground truth, used only for scoring (and, in data-collection flights, for guidance). -->
      <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
        <dimensions>3</dimensions>
      </plugin>
    </model>'''


def box(name, x, y, sx, sy, sz, yaw=0.0, color='0.6 0.6 0.65 1'):
    return f'''
        <collision name="{name}_c"><pose>{x} {y} {sz / 2} 0 0 {yaw}</pose><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>
        <visual name="{name}_v"><pose>{x} {y} {sz / 2} 0 0 {yaw}</pose><geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
          <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>'''


def tree(name, x, y, h=6.0):
    return f'''
        <visual name="{name}_trunk"><pose>{x} {y} {h * 0.25} 0 0 0</pose><geometry><cylinder><radius>0.25</radius><length>{h * 0.5}</length></cylinder></geometry>
          <material><ambient>0.4 0.27 0.15 1</ambient><diffuse>0.4 0.27 0.15 1</diffuse></material></visual>
        <visual name="{name}_crown"><pose>{x} {y} {h * 0.7} 0 0 0</pose><geometry><sphere><radius>{h * 0.25}</radius></sphere></geometry>
          <material><ambient>0.15 0.45 0.18 1</ambient><diffuse>0.15 0.45 0.18 1</diffuse></material></visual>
        <collision name="{name}_c"><pose>{x} {y} {h * 0.5} 0 0 0</pose><geometry><cylinder><radius>0.3</radius><length>{h}</length></cylinder></geometry></collision>'''


def world(name, drone_xml, scenery_xml, overhead_pose, overhead_fov=1.15, ground_color='0.42 0.55 0.35 1',
          lat=0.35, lon=32.58):
    return f'''<?xml version="1.0" ?>
<!-- Generated by common/make_world.py -->
<sdf version="1.9">
  <world name="{name}">
    <physics name="4ms" type="ignored"><max_step_size>0.004</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model><world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>{lat}</latitude_deg><longitude_deg>{lon}</longitude_deg><elevation>0</elevation><heading_deg>0</heading_deg>
    </spherical_coordinates>
    <scene><ambient>0.6 0.6 0.6 1</ambient><background>0.65 0.78 0.92 1</background><shadows>false</shadows></scene>
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows><pose>0 0 50 0 0 0</pose>
      <diffuse>0.95 0.95 0.9 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.4 0.2 -0.9</direction>
    </light>
    <model name="ground_plane"><static>true</static>
      <link name="link">
        <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>400 400</size></plane></geometry></collision>
        <visual name="visual"><geometry><plane><normal>0 0 1</normal><size>400 400</size></plane></geometry>
          <material><ambient>{ground_color}</ambient><diffuse>{ground_color}</diffuse></material></visual>
      </link>
    </model>
    <model name="scenery"><static>true</static><link name="link">{scenery_xml}
    </link></model>
    <model name="overhead_cam"><static>true</static><pose>{overhead_pose}</pose>
      <link name="link">{camera('overhead_cam', '0 0 0 0 0 0', overhead_fov, 400)}
      </link>
    </model>
{drone_xml}
  </world>
</sdf>
'''
