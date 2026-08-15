"""
Maps the CNN's 22 CyberGlove-channel angle outputs (degrees) to MyoSuite's MyoHand MuJoCo
joint positions (radians, qpos array).

Verified directly against a loaded 'myoHandPoseFixed-v0' environment (2026-07-13): 23 qpos
joints with real anatomical names (mcp2_flexion, pm3_flexion, cmc_abduction, etc.), 39 muscle
actuators. This mapping targets qpos directly (kinematic/position control) -- setting the
joint angle and calling mj_forward(), not driving the 39 muscles through mj_step(). That's
the deliberately crude first version: prove the CNN->hand loop works before adding a muscle-
activation tracking controller for real contact physics.

Finger numbering in MyoHand follows anatomical convention: 2=index, 3=middle, 4=ring,
5=little -- matches our CyberGlove channel names directly.
"""
import numpy as np

GLOVE_CHANNEL_NAMES = [
    'ThumbRot', 'ThumbMPJ', 'ThumbIJ', 'ThumbAbd',
    'IndexMPJ', 'IndexPIJ', 'IndexDIJ', 'IndexAbd',
    'MiddleMPJ', 'MiddlePIJ', 'MiddleDIJ', 'PalmArch',
    'RingMPJ', 'RingPIJ', 'RingDIJ', 'RingAbd',
    'LittleMPJ', 'LittlePIJ', 'LittleDIJ', 'LittleAbd',
    'WristPitch', 'WristYaw',
]
CH = {name: i for i, name in enumerate(GLOVE_CHANNEL_NAMES)}

# Known data quirk (same one found in every prior notebook): MiddleDIJ has a raw-value scale
# ~10x wider than every other channel across every subject checked -- a calibration artifact
# for that one sensor, not a real angle. The per-joint radian clip below (from MyoHand's own
# jnt_range) catches this automatically since it's physically impossible for md3_flexion to
# exceed ~90 degrees, but it's called out explicitly here so it's not a silent surprise.
ANOMALOUS_CHANNEL = 'MiddleDIJ'

# glove_channel_name -> myohand_joint_name. Joints with no direct glove sensor (pro_sup,
# mcp3_abduction -- middle finger has no abduction sensor on the glove either) are left
# unmapped and stay at 0 (neutral pose).
GLOVE_TO_MYOHAND_JOINT = {
    'WristPitch':  'flexion',
    'WristYaw':    'deviation',
    'ThumbAbd':    'cmc_abduction',
    'ThumbRot':    'cmc_flexion',
    'ThumbMPJ':    'mp_flexion',
    'ThumbIJ':     'ip_flexion',
    'IndexMPJ':    'mcp2_flexion',
    'IndexAbd':    'mcp2_abduction',
    'IndexPIJ':    'pm2_flexion',
    'IndexDIJ':    'md2_flexion',
    'MiddleMPJ':   'mcp3_flexion',
    'MiddlePIJ':   'pm3_flexion',
    'MiddleDIJ':   'md3_flexion',
    'RingMPJ':     'mcp4_flexion',
    'RingAbd':     'mcp4_abduction',
    'RingPIJ':     'pm4_flexion',
    'RingDIJ':     'md4_flexion',
    'LittleMPJ':   'mcp5_flexion',
    'LittleAbd':   'mcp5_abduction',
    'LittlePIJ':   'pm5_flexion',
    'LittleDIJ':   'md5_flexion',
}


def build_joint_index_map(mj_model):
    """Resolve joint names to qpos indices once, against the actual loaded model -- don't
    assume a fixed index order, MuJoCo doesn't guarantee it stays the same across versions."""
    name_to_qpos_idx = {}
    for i in range(mj_model.nq):
        name_to_qpos_idx[mj_model.joint(i).name] = i

    mapping = {}
    for glove_name, joint_name in GLOVE_TO_MYOHAND_JOINT.items():
        if joint_name not in name_to_qpos_idx:
            print(f'WARNING: joint "{joint_name}" (for {glove_name}) not found in this MuJoCo model -- skipping')
            continue
        qpos_idx = name_to_qpos_idx[joint_name]
        jnt_range = mj_model.jnt_range[qpos_idx]
        mapping[glove_name] = {'qpos_idx': qpos_idx, 'range_rad': (float(jnt_range[0]), float(jnt_range[1]))}
    return mapping


def apply_glove_row_to_qpos(glove_row, qpos, joint_index_map):
    """glove_row: length-22 array of predicted angles in degrees. Mutates qpos in place."""
    for glove_name, info in joint_index_map.items():
        angle_deg = glove_row[CH[glove_name]]
        angle_rad = np.radians(angle_deg)
        lo, hi = info['range_rad']
        qpos[info['qpos_idx']] = np.clip(angle_rad, lo, hi)
