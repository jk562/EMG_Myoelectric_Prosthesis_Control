// JS port of the old bone_mapping.py -- same 22-channel layout, same Mixamo bone names,
// same "extra tip bone has no sensor, follows previous segment damped" logic. The rotation
// axis default (local X, positive = flexion) was verified by direct visual testing in
// Blender against this exact skeleton -- carries over here since it's the same underlying
// bone data, just re-exported to glTF. Still worth a quick visual sanity check once loaded,
// the same way the Blender version needed one, since the FBX->glTF coordinate conversion
// (Z-up to Y-up) is a place a subtle mismatch could in principle show up.

export const GLOVE_CHANNEL_NAMES = [
  'ThumbRot', 'ThumbMPJ', 'ThumbIJ', 'ThumbAbd',
  'IndexMPJ', 'IndexPIJ', 'IndexDIJ', 'IndexAbd',
  'MiddleMPJ', 'MiddlePIJ', 'MiddleDIJ', 'PalmArch',
  'RingMPJ', 'RingPIJ', 'RingDIJ', 'RingAbd',
  'LittleMPJ', 'LittlePIJ', 'LittleDIJ', 'LittleAbd',
  'WristPitch', 'WristYaw',
];
export const CH = Object.fromEntries(GLOVE_CHANNEL_NAMES.map((name, i) => [name, i]));

// Known data quirk (see Test2/Test3 notes): MiddleDIJ has a raw-value scale ~10x wider than
// every other channel across every subject checked -- a calibration artifact, not a real
// angle. Clipped so it can't visibly break the middle finger's tip joint.
const ANOMALOUS_CHANNEL_CLIP = { MiddleDIJ: [0, 90] };

const MIXAMO_FINGER_BONES = {
  Thumb: ['Thumb1', 'Thumb2', 'Thumb3', 'Thumb4'],
  Index: ['Index1', 'Index2', 'Index3', 'Index4'],
  Middle: ['Middle1', 'Middle2', 'Middle3', 'Middle4'],
  Ring: ['Ring1', 'Ring2', 'Ring3', 'Ring4'],
  Little: ['Pinky1', 'Pinky2', 'Pinky3', 'Pinky4'],   // Mixamo calls it "Pinky"
};

// Thumb only has 2 real flex sensors (MPJ, IJ); non-thumb fingers have 3 (MCP/PIP/DIP).
// The remaining tip bone(s) have no sensor and just follow the previous segment, damped.
const FINGER_CHANNEL_MAP = {
  Thumb: ['ThumbMPJ', 'ThumbIJ', null, null],
  Index: ['IndexMPJ', 'IndexPIJ', 'IndexDIJ', null],
  Middle: ['MiddleMPJ', 'MiddlePIJ', 'MiddleDIJ', null],
  Ring: ['RingMPJ', 'RingPIJ', 'RingDIJ', null],
  Little: ['LittleMPJ', 'LittlePIJ', 'LittleDIJ', null],
};

export function boneName(finger, segIdx, side = 'Right') {
  return `mixamorig:${side}Hand${MIXAMO_FINGER_BONES[finger][segIdx]}`;
}

// Per-bone axis/sign overrides, filled in only if a visual check shows something is wrong
// for a specific bone. Empty by default -- same convention as bone_mapping.py.
export const AXIS_OVERRIDES = {};
const DEFAULT_AXIS = 'X';
const DEFAULT_SIGN = 1;

export function gloveRowToBoneRotations(gloveRow, side = 'Right') {
  const rotations = {};

  for (const [finger, channels] of Object.entries(FINGER_CHANNEL_MAP)) {
    let prevAngle = 0.0;
    channels.forEach((chanName, segIdx) => {
      const bname = boneName(finger, segIdx, side);
      let angle;
      if (chanName === null) {
        angle = prevAngle * 0.5;
      } else {
        angle = gloveRow[CH[chanName]];
        if (ANOMALOUS_CHANNEL_CLIP[chanName]) {
          const [lo, hi] = ANOMALOUS_CHANNEL_CLIP[chanName];
          angle = Math.max(lo, Math.min(hi, angle));
        }
        prevAngle = angle;
      }
      const boneLabel = MIXAMO_FINGER_BONES[finger][segIdx];
      const [axis, sign] = AXIS_OVERRIDES[boneLabel] || [DEFAULT_AXIS, DEFAULT_SIGN];
      rotations[bname] = { axis, angleDeg: sign * angle };
    });
  }

  rotations[`mixamorig:${side}Hand`] = { axis: 'Z', angleDeg: gloveRow[CH['WristYaw']] };
  rotations[`mixamorig:${side}ForeArm`] = { axis: 'X', angleDeg: gloveRow[CH['WristPitch']] };

  return rotations;
}
