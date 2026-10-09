"""This code defines a set of models and constants to model heat exchange and blood flow
in different body parts and layers."""

import numpy as np

from pythermalcomfort.classes_return import JOS3BodyParts
from pythermalcomfort.jos3_functions.parameters import Default

LAYER_NAMES = ["artery", "vein", "sfvein", "core", "muscle", "fat", "skin"]


def index_order():
    """Define the index's order of the matrix.

    Returns
    -------
    index_dict : nested dictionary
        keys are the body names and layer names
    """
    # Defines existing layers as 1 or None
    index_dict = {}

    for key in ["head", "pelvis"]:
        index_dict[key] = {
            "artery": 1,
            "vein": 1,
            "sfvein": None,
            "core": 1,
            "muscle": 1,
            "fat": 1,
            "skin": 1,
        }

    for key in ["neck", "chest", "back"]:
        index_dict[key] = {
            "artery": 1,
            "vein": 1,
            "sfvein": None,
            "core": 1,
            "muscle": None,
            "fat": None,
            "skin": 1,
        }

    for key in JOS3BodyParts.get_attribute_names()[5:]:  # limb segments
        index_dict[key] = {
            "artery": 1,
            "vein": 1,
            "sfvein": 1,
            "core": 1,
            "muscle": None,
            "fat": None,
            "skin": 1,
        }

    # Sets ordered indices in the matrix
    index_dict["CB"] = 0
    order_count = 1
    for bn in JOS3BodyParts.get_attribute_names():
        for ln in LAYER_NAMES:
            if index_dict[bn][ln] is not None:
                index_dict[bn][ln] = order_count
                order_count += 1

    return index_dict, order_count


IDICT, NUM_NODES = index_order()


def index_by_layer(layer):
    """Get indices of the matrix by the layer name.

    Parameters
    ----------
    layer : str
        Layer name of jos.
        ex) artery, vein, sfvein, core, muscle, fat or skin.

    Returns
    -------
    indices of the matrix : list
    """
    # Gets indices by the layer name
    out_index = []
    for bn in JOS3BodyParts.get_attribute_names():
        for ln in LAYER_NAMES:
            if (layer.lower() == ln) and IDICT[bn][ln]:
                out_index.append(IDICT[bn][ln])
    return out_index


def valid_index_by_layer(layer):
    """Get indices of the matrix by the layer name.

    Parameters
    ----------
    layer : str
        Layer name of jos.
        ex) artery, vein, sfvein, core, muscle, fat or skin.

    Returns
    -------
    indices of the matrix : list
    """
    # Gets valid indices of the layer name
    out_index = []
    for i, bn in enumerate(JOS3BodyParts.get_attribute_names()):
        if IDICT[bn][layer]:
            out_index.append(i)
    return out_index


# Constant parameters of the matrix' indicies
INDEX = {}
VINDEX = {}
for key in LAYER_NAMES:
    INDEX[key] = index_by_layer(key)
    VINDEX[key] = valid_index_by_layer(key)


def local_arr(bf_core, bf_muscle, bf_fat, bf_skin, bf_ava_hand, bf_ava_foot):
    """Create matrix to calculate heat exchange by blood flow in each segment.

    [W/K]

    1.067 [Wh/(L･K)] * Bloodflow [L/h] = [W/K]
    """
    bf_local = np.zeros((NUM_NODES, NUM_NODES))
    for i, bn in enumerate(JOS3BodyParts.get_attribute_names()):
        # Dictionary of indecies in each body segment
        # key = layer name, value = index of matrix
        index_of = IDICT[bn]

        # Common
        bf_local[index_of["core"], index_of["artery"]] = 1.067 * bf_core[i]  # art to cr
        bf_local[index_of["skin"], index_of["artery"]] = 1.067 * bf_skin[i]  # art to sk
        bf_local[index_of["vein"], index_of["core"]] = 1.067 * bf_core[i]  # vein to cr
        bf_local[index_of["vein"], index_of["skin"]] = 1.067 * bf_skin[i]  # vein to sk

        # If the segment has a muslce or fat layer
        if index_of["muscle"] is not None:
            bf_local[index_of["muscle"], index_of["artery"]] = (
                1.067 * bf_muscle[i]
            )  # art to ms
            bf_local[index_of["vein"], index_of["muscle"]] = (
                1.067 * bf_muscle[i]
            )  # vein to ms
        if index_of["fat"] is not None:
            bf_local[index_of["fat"], index_of["artery"]] = (
                1.067 * bf_fat[i]
            )  # art to fat
            bf_local[index_of["vein"], index_of["fat"]] = (
                1.067 * bf_fat[i]
            )  # vein to fat

        # Only hand
        if i == 7 or i == 10:
            bf_local[index_of["sfvein"], index_of["artery"]] = (
                1.067 * bf_ava_hand
            )  # art to sfvein
        # Only foot
        if i == 13 or i == 16:
            bf_local[index_of["sfvein"], index_of["artery"]] = (
                1.067 * bf_ava_foot
            )  # art to sfvein

    return bf_local


def vessel_blood_flow(bf_core, bf_muscle, bf_fat, bf_skin, bf_ava_hand, bf_ava_foot):
    """Get artery and vein blood flow rate [l/h]."""
    xbf = bf_core + bf_muscle + bf_fat + bf_skin

    bf_art = np.zeros(Default.num_body_parts)
    bf_vein = np.zeros(Default.num_body_parts)

    # head
    bf_art[0] = xbf[0]
    bf_vein[0] = xbf[0]

    # neck (+head)
    bf_art[1] = xbf[1] + xbf[0]
    bf_vein[1] = xbf[1] + xbf[0]

    # chest
    bf_art[2] = xbf[2]
    bf_vein[2] = xbf[2]

    # back
    bf_art[3] = xbf[3]
    bf_vein[3] = xbf[3]

    # pelvis (+Thighs, Legs, Feet, AVA_Feet)
    bf_art[4] = xbf[4] + xbf[11 : Default.num_body_parts].sum() + 2 * bf_ava_foot
    bf_vein[4] = xbf[4] + xbf[11 : Default.num_body_parts].sum() + 2 * bf_ava_foot

    # L.Shoulder (+Arm, Hand, (arteryのみAVA_Hand))
    bf_art[5] = xbf[5:8].sum() + bf_ava_hand
    bf_vein[5] = xbf[5:8].sum()

    # L.Arm (+Hand)
    bf_art[6] = xbf[6:8].sum() + bf_ava_hand
    bf_vein[6] = xbf[6:8].sum()

    # L.Hand
    bf_art[7] = xbf[7] + bf_ava_hand
    bf_vein[7] = xbf[7]

    # R.Shoulder (+Arm, Hand, (arteryのみAVA_Hand))
    bf_art[8] = xbf[8:11].sum() + bf_ava_hand
    bf_vein[8] = xbf[8:11].sum()

    # R.Arm (+Hand)
    bf_art[9] = xbf[9:11].sum() + bf_ava_hand
    bf_vein[9] = xbf[9:11].sum()

    # R.Hand
    bf_art[10] = xbf[10] + bf_ava_hand
    bf_vein[10] = xbf[10]

    # L.Thigh (+Leg, Foot, (arteryのみAVA_Foot))
    bf_art[11] = xbf[11:14].sum() + bf_ava_foot
    bf_vein[11] = xbf[11:14].sum()

    # L.Leg (+Foot)
    bf_art[12] = xbf[12:14].sum() + bf_ava_foot
    bf_vein[12] = xbf[12:14].sum()

    # L.Foot
    bf_art[13] = xbf[13] + bf_ava_foot
    bf_vein[13] = xbf[13]

    # R.Thigh (+Leg, Foot, (arteryのみAVA_Foot))
    bf_art[14] = xbf[14 : Default.num_body_parts].sum() + bf_ava_foot
    bf_vein[14] = xbf[14 : Default.num_body_parts].sum()

    # R.Leg (+Foot)
    bf_art[15] = xbf[15 : Default.num_body_parts].sum() + bf_ava_foot
    bf_vein[15] = xbf[15 : Default.num_body_parts].sum()

    # R.Foot
    bf_art[16] = xbf[16] + bf_ava_foot
    bf_vein[16] = xbf[16]

    return bf_art, bf_vein


def whole_body(bf_art, bf_vein, bf_ava_hand, bf_ava_foot):
    """Create matrix to calculate heat exchange by blood flow between segments.

    [W/K]
    """

    def flow(up, down, bloodflow):
        arr = np.zeros((NUM_NODES, NUM_NODES))
        arr[down, up] = (
            1.067 * bloodflow
        )  # Coefficient = 1.067 [Wh/L.K] Change units [L/h] to [W/K]
        return arr

    arr83 = np.zeros((NUM_NODES, NUM_NODES))
    # Matrix offsets of segments
    CB = IDICT["CB"]
    head = IDICT["head"]["artery"]
    neck = IDICT["neck"]["artery"]
    chest = IDICT["chest"]["artery"]
    back = IDICT["back"]["artery"]
    pelvis = IDICT["pelvis"]["artery"]
    left_shoulder = IDICT["left_shoulder"]["artery"]
    left_arm = IDICT["left_arm"]["artery"]
    left_hand = IDICT["left_hand"]["artery"]
    right_shoulder = IDICT["right_shoulder"]["artery"]
    right_arm = IDICT["right_arm"]["artery"]
    right_hand = IDICT["right_hand"]["artery"]
    left_thigh = IDICT["left_thigh"]["artery"]
    left_leg = IDICT["left_leg"]["artery"]
    left_foot = IDICT["left_foot"]["artery"]
    right_thigh = IDICT["right_thigh"]["artery"]
    right_leg = IDICT["right_leg"]["artery"]
    right_foot = IDICT["right_foot"]["artery"]

    arr83 += flow(up=CB, down=neck, bloodflow=bf_art[1])  # CB to neck.art
    arr83 += flow(up=neck, down=head, bloodflow=bf_art[0])  # neck.art to head.art
    arr83 += flow(
        up=head + 1, down=neck + 1, bloodflow=bf_vein[0]
    )  # head.vein to neck.vein
    arr83 += flow(up=neck + 1, down=CB, bloodflow=bf_vein[1])  # neck.vein to CB

    arr83 += flow(up=CB, down=chest, bloodflow=bf_art[2])  # CB to chest.art
    arr83 += flow(up=chest + 1, down=CB, bloodflow=bf_vein[2])  # chest.vein to CB

    arr83 += flow(up=CB, down=back, bloodflow=bf_art[3])  # CB to back.art
    arr83 += flow(up=back + 1, down=CB, bloodflow=bf_vein[3])  # back.vein to CB

    arr83 += flow(up=CB, down=pelvis, bloodflow=bf_art[4])  # CB to pelvis.art
    arr83 += flow(up=pelvis + 1, down=CB, bloodflow=bf_vein[4])  # pelvis.vein to CB

    arr83 += flow(
        up=CB, down=left_shoulder, bloodflow=bf_art[5]
    )  # CB to left_shoulder.art
    arr83 += flow(
        up=left_shoulder,
        down=left_arm,
        bloodflow=bf_art[6],
    )  # left_shoulder.art to left_arm.art
    arr83 += flow(
        up=left_arm, down=left_hand, bloodflow=bf_art[7]
    )  # left_arm.art to left_hand.art
    arr83 += flow(
        up=left_hand + 1,
        down=left_arm + 1,
        bloodflow=bf_vein[7],
    )  # left_hand.vein to left_arm.vein
    arr83 += flow(
        up=left_arm + 1,
        down=left_shoulder + 1,
        bloodflow=bf_vein[6],
    )  # left_arm.vein to left_shoulder.vein
    arr83 += flow(
        up=left_shoulder + 1, down=CB, bloodflow=bf_vein[5]
    )  # left_shoulder.vein to CB
    arr83 += flow(
        up=left_hand + 2,
        down=left_arm + 2,
        bloodflow=bf_ava_hand,
    )  # left_hand.sfvein to left_arm.sfvein
    arr83 += flow(
        up=left_arm + 2,
        down=left_shoulder + 2,
        bloodflow=bf_ava_hand,
    )  # left_arm.sfvein to left_shoulder.sfvein
    arr83 += flow(
        up=left_shoulder + 2, down=CB, bloodflow=bf_ava_hand
    )  # left_shoulder.sfvein to CB

    arr83 += flow(
        up=CB, down=right_shoulder, bloodflow=bf_art[8]
    )  # CB to right_shoulder.art
    arr83 += flow(
        up=right_shoulder,
        down=right_arm,
        bloodflow=bf_art[9],
    )  # right_shoulder.art to right_arm.art
    arr83 += flow(
        up=right_arm, down=right_hand, bloodflow=bf_art[10]
    )  # right_arm.art to right_hand.art
    arr83 += flow(
        up=right_hand + 1,
        down=right_arm + 1,
        bloodflow=bf_vein[10],
    )  # right_hand.vein to right_arm.vein
    arr83 += flow(
        up=right_arm + 1,
        down=right_shoulder + 1,
        bloodflow=bf_vein[9],
    )  # right_arm.vein to right_shoulder.vein
    arr83 += flow(
        up=right_shoulder + 1, down=CB, bloodflow=bf_vein[8]
    )  # right_shoulder.vein to CB
    arr83 += flow(
        up=right_hand + 2,
        down=right_arm + 2,
        bloodflow=bf_ava_hand,
    )  # right_hand.sfvein to right_arm.sfvein
    arr83 += flow(
        up=right_arm + 2,
        down=right_shoulder + 2,
        bloodflow=bf_ava_hand,
    )  # right_arm.sfvein to right_shoulder.sfvein
    arr83 += flow(
        up=right_shoulder + 2, down=CB, bloodflow=bf_ava_hand
    )  # right_shoulder.sfvein to CB

    arr83 += flow(
        up=pelvis, down=left_thigh, bloodflow=bf_art[11]
    )  # pelvis to left_thigh.art
    arr83 += flow(
        up=left_thigh, down=left_leg, bloodflow=bf_art[12]
    )  # left_thigh.art to left_leg.art
    arr83 += flow(
        up=left_leg, down=left_foot, bloodflow=bf_art[13]
    )  # left_leg.art to left_foot.art
    arr83 += flow(
        up=left_foot + 1,
        down=left_leg + 1,
        bloodflow=bf_vein[13],
    )  # left_foot.vein to left_leg.vein
    arr83 += flow(
        up=left_leg + 1,
        down=left_thigh + 1,
        bloodflow=bf_vein[12],
    )  # left_leg.vein to left_thigh.vein
    arr83 += flow(
        up=left_thigh + 1, down=pelvis + 1, bloodflow=bf_vein[11]
    )  # left_thigh.vein to pelvis
    arr83 += flow(
        up=left_foot + 2,
        down=left_leg + 2,
        bloodflow=bf_ava_foot,
    )  # left_foot.sfvein to left_leg.sfvein
    arr83 += flow(
        up=left_leg + 2,
        down=left_thigh + 2,
        bloodflow=bf_ava_foot,
    )  # left_leg.sfvein to left_thigh.sfvein
    arr83 += flow(
        up=left_thigh + 2, down=pelvis + 1, bloodflow=bf_ava_foot
    )  # left_thigh.vein to pelvis

    arr83 += flow(
        up=pelvis, down=right_thigh, bloodflow=bf_art[14]
    )  # pelvis to right_thigh.art
    arr83 += flow(
        up=right_thigh,
        down=right_leg,
        bloodflow=bf_art[15],
    )  # right_thigh.art to right_leg.art
    arr83 += flow(
        up=right_leg, down=right_foot, bloodflow=bf_art[16]
    )  # right_leg.art to right_foot.art
    arr83 += flow(
        up=right_foot + 1,
        down=right_leg + 1,
        bloodflow=bf_vein[16],
    )  # right_foot.vein to right_leg.vein
    arr83 += flow(
        up=right_leg + 1,
        down=right_thigh + 1,
        bloodflow=bf_vein[15],
    )  # right_leg.vein to right_thigh.vein
    arr83 += flow(
        up=right_thigh + 1,
        down=pelvis + 1,
        bloodflow=bf_vein[14],
    )  # right_thigh.vein to pelvis
    arr83 += flow(
        up=right_foot + 2,
        down=right_leg + 2,
        bloodflow=bf_ava_foot,
    )  # right_foot.sfvein to right_leg.sfvein
    arr83 += flow(
        up=right_leg + 2,
        down=right_thigh + 2,
        bloodflow=bf_ava_foot,
    )  # right_leg.sfvein to right_thigh.sfvein
    arr83 += flow(
        up=right_thigh + 2,
        down=pelvis + 1,
        bloodflow=bf_ava_foot,
    )  # right_thigh.vein to pelvis

    return arr83


def remove_body_name(text):
    """Remove the body name from the parameter name.

    Parameters
    ----------
    text : str
        Parameter name

    Returns
    -------
    rtext : str
        Parameter name removed the body name.
    removed : str
        The removed body name
    """
    rtext = text
    removed = None

    # Remove the body part name from the parameter name
    for bn in JOS3BodyParts.get_attribute_names():
        if bn in text:
            rtext = rtext.replace(
                bn,
                "",
            )  # Remove the body part name from the parameter name
            rtext = rtext.removesuffix("_")  # Remove the trailing underscore
            removed = bn  # Store the removed body part name
            break
    return rtext, removed
