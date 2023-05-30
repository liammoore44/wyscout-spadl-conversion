"""Wyscout event stream data to SPADL converter."""
from typing import Any, Dict, List, Optional, Set, cast

import numpy as np
import pandas as pd  # type: ignore
from pandera.typing import DataFrame

from . import config as spadlconfig
from .base import (
    _add_dribbles,
    _fix_clearances,
    _fix_direction_of_play,
    min_dribble_length,
)
from .schema import SPADLSchema

###################################
# WARNING: HERE BE DRAGONS
# This code for converting wyscout data was organically grown over a long period of time.
# It works for now, but needs to be cleaned up in the future.
# Enter at your own risk.
###################################


def convert_to_actions(events: pd.DataFrame, home_team_id: int) -> DataFrame[SPADLSchema]:
    """
    Convert Wyscout events to SPADL actions.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame containing Wyscout events from a single game.
    home_team_id : int
        ID of the home team in the corresponding game.

    Returns
    -------
    actions : pd.DataFrame
        DataFrame with corresponding SPADL actions.

    """
    events["period_id"] = events.matchPeriod.apply(lambda x: wyscout_periods[x])
    events["milliseconds"] = (events.second + (events.minute * 60)) * 1000

    events = make_new_positions(events)
    events_fixed = fix_wyscout_events(events)
    actions = create_df_actions(events_fixed)
    actions = fix_actions(actions)
    actions = _fix_direction_of_play(actions, home_team_id)
    actions = _fix_clearances(actions)
    actions["action_id"] = range(len(actions))
    actions = _add_dribbles(actions)

    return cast(DataFrame[SPADLSchema], actions)


def _get_tag_set(tags: List[Dict[str, Any]]) -> Set[int]:
    return {tag["id"] for tag in tags}


def get_tagsdf(events: pd.DataFrame) -> pd.DataFrame:
    """Represent Wyscout tags as a boolean dataframe.

    Parameters
    ----------
    events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.DataFrame
        A dataframe with a column for each tag.
    """
    tags = events.tags.apply(_get_tag_set)
    tagsdf = pd.DataFrame()
    for tag_id, column in wyscout_tags:
        tagsdf[column] = tags.apply(lambda x, tag=tag_id: tag in x)
    return tagsdf


wyscout_tags = [
    (101, "goal"),
    (102, "own_goal"),
    (301, "assist"),
    (302, "key_pass"),
    (1901, "counter_attack"),
    (401, "left_foot"),
    (402, "right_foot"),
    (403, "head/body"),
    (1101, "direct"),
    (1102, "indirect"),
    (2001, "dangerous_ball_lost"),
    (2101, "blocked"),
    (801, "high"),
    (802, "low"),
    (1401, "interception"),
    (1501, "clearance"),
    (201, "opportunity"),
    (1301, "feint"),
    (1302, "missed_ball"),
    (501, "free_space_right"),
    (502, "free_space_left"),
    (503, "take_on_left"),
    (504, "take_on_right"),
    (1601, "sliding_tackle"),
    (601, "anticipated"),
    (602, "anticipation"),
    (1701, "red_card"),
    (1702, "yellow_card"),
    (1703, "second_yellow_card"),
    (1201, "position_goal_low_center"),
    (1202, "position_goal_low_right"),
    (1203, "position_goal_mid_center"),
    (1204, "position_goal_mid_left"),
    (1205, "position_goal_low_left"),
    (1206, "position_goal_mid_right"),
    (1207, "position_goal_high_center"),
    (1208, "position_goal_high_left"),
    (1209, "position_goal_high_right"),
    (1210, "position_out_low_right"),
    (1211, "position_out_mid_left"),
    (1212, "position_out_low_left"),
    (1213, "position_out_mid_right"),
    (1214, "position_out_high_center"),
    (1215, "position_out_high_left"),
    (1216, "position_out_high_right"),
    (1217, "position_post_low_right"),
    (1218, "position_post_mid_left"),
    (1219, "position_post_low_left"),
    (1220, "position_post_mid_right"),
    (1221, "position_post_high_center"),
    (1222, "position_post_high_left"),
    (1223, "position_post_high_right"),
    (901, "through"),
    (1001, "fairplay"),
    (701, "lost"),
    (702, "neutral"),
    (703, "won"),
    (1801, "accurate"),
    (1802, "not_accurate"),
]


def _make_position_vars(events: pd.DataFrame) -> pd.DataFrame:
    # Join events with related events
    related_events = events.merge(events, left_on='relatedEventId', right_on='id', suffixes=['', '_related_event'], how='left')
    # Check that the related event came after the event
    related_event_mask = ((related_events.id_related_event is not None)
                          & (related_events.matchTimestamp <= related_events.matchTimestamp_related_event))
    # Set end of event to the start position of its related event
    related_events.loc[related_event_mask, "end_x"] = related_events.location_x_related_event
    related_events.loc[related_event_mask, "end_y"] = related_events.location_y_related_event
    return related_events


def make_new_positions(events: pd.DataFrame) -> pd.DataFrame:
    """Extract the start and end coordinates for each action.

    Parameters
    ----------
    events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.DataFrame
        Wyscout event dataframe with start and end coordinates for each action.
    """
    # Initialise start and end values
    events["start_x"] = events.location_x
    events["start_y"] = events.location_y
    events["end_x"] = events.location_x
    events["end_y"] = events.location_y
    events = _make_position_vars(events)
    # Update start and end values where possible
    events.loc[events.pass_accurate.notna(), "end_x"] = events.pass_endLocation_x
    events.loc[events.pass_accurate.notna(), "end_y"] = events.pass_endLocation_y

    events.loc[events.carry_progression.notna(), "end_x"] = events.carry_endLocation_x
    events.loc[events.carry_progression.notna(), "end_y"] = events.carry_endLocation_y
    return events


def fix_wyscout_events(df_events: pd.DataFrame) -> pd.DataFrame:
    """Perform some fixes on the Wyscout events such that the spadl action dataframe can be built.

    Parameters
    ----------
    df_events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.DataFrame
        Wyscout event dataframe with an extra column 'offside'
    """
    df_events = create_shot_coordinates(df_events)
    df_events = convert_duels(df_events)
    df_events = insert_interception_passes(df_events)
    df_events = add_offside_variable(df_events)
    df_events = convert_touches(df_events)
    df_events = convert_simulations(df_events)
    return df_events


def create_shot_coordinates(df_events: pd.DataFrame) -> pd.DataFrame:
    """Create shot coordinates (estimates) from the Wyscout tags.

    Parameters
    ----------
    df_events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.DataFrame
        Wyscout event dataframe with end coordinates for shots
    """
    goal_center_idx = (
        (df_events.shot_goalZone=='gc')
        | (df_events.shot_goalZone=='gb')
        | (df_events.shot_goalZone=='gt')
        | (df_events.shot_goalZone=='ot')
        | (df_events.shot_goalZone=='pt')
                       
    )
    df_events.loc[goal_center_idx, "end_x"] = 100.0
    df_events.loc[goal_center_idx, "end_y"] = 50.0

    goal_right_idx = (
        (df_events.shot_goalZone=='gtr')
        | (df_events.shot_goalZone=='gr')
        | (df_events.shot_goalZone=='gbr')
                       
    )
    df_events.loc[goal_right_idx, "end_x"] = 100.0
    df_events.loc[goal_right_idx, "end_y"] = 55.0

    goal_left_idx = (
        (df_events.shot_goalZone=='gtl')
        | (df_events.shot_goalZone=='gl')
        | (df_events.shot_goalZone=='glb')
                       
    )
    df_events.loc[goal_left_idx, "end_x"] = 100.0
    df_events.loc[goal_left_idx, "end_y"] = 45.0

    out_right_idx = (
        (df_events.shot_goalZone=='otr')
        | (df_events.shot_goalZone=='or')
        | (df_events.shot_goalZone=='obr')    
    )
    df_events.loc[out_right_idx, "end_x"] = 100.0
    df_events.loc[out_right_idx, "end_y"] = 60.0

    out_left_idx = (
       (df_events.shot_goalZone == 'olb')
        | (df_events.shot_goalZone == 'ol')
        | (df_events.shot_goalZone == 'otl')
    )
    df_events.loc[out_left_idx, "end_x"] = 100.0
    df_events.loc[out_left_idx, "end_y"] = 40.0

    post_left_idx = (
        (df_events.shot_goalZone == 'ptl')
        | (df_events.shot_goalZone == 'pl')
        | (df_events.shot_goalZone == 'plb')
    )
    df_events.loc[post_left_idx, "end_x"] = 100.0
    df_events.loc[post_left_idx, "end_y"] = 55.38

    post_right_idx = (
        (df_events.shot_goalZone == 'ptr')
        | (df_events.shot_goalZone == 'pr')
        | (df_events.shot_goalZone == 'pbr')
    )
    df_events.loc[post_right_idx, "end_x"] = 100.0
    df_events.loc[post_right_idx, "end_y"] = 44.62

    blocked_idx = (df_events.shot_goalZone == "bc")
    df_events.loc[blocked_idx, "end_x"] = df_events.loc[blocked_idx, "start_x"]
    df_events.loc[blocked_idx, "end_y"] = df_events.loc[blocked_idx, "start_y"]

    return df_events


def convert_duels(events_df: pd.DataFrame) -> pd.DataFrame:
    """Convert duel events.

    This function converts Wyscout duels that end with the ball out of field
    (subtype_id 50) into a pass for the player winning the duel to the location
    of where the ball went out of field. The remaining duels are removed as
    they are not on-the-ball actions.

    Parameters
    ----------
    df_events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.DataFrame
        Wyscout event dataframe in which the duels are either removed or
        transformed into a pass
    """
    df_events = events_df.copy()
    # Shift events dataframe by one and two time steps
    df_events1 = df_events.shift(-1)
    df_events2 = df_events.shift(-2)

    # Define selector for same period id
    selector_same_period = df_events["period_id"] == df_events2["period_id"]

    # Define selector for duels that are followed by an 'out of field' event
    selector_duel_out_of_field = (
        (df_events.type_primary == 'duel')
        & (df_events1.type_primary == 'duel')
        & (df_events2.type_secondary.apply(lambda x: x is not None and "ball_out" in x))
        & selector_same_period
    )

    # Define selectors for current time step
    selector0_duel_won = selector_duel_out_of_field & (
        df_events["team_id"] != df_events2["team_id"]
    )
    selector0_duel_won_air = selector0_duel_won & (df_events1.type_secondary.apply(lambda x: x is not None and "aerial_duel" in x))
    selector0_duel_won_not_air = selector0_duel_won & (df_events1.type_secondary.apply(lambda x: x is not None and "aerial_duel" not in x))

    # Define selectors for next time step
    selector1_duel_won = selector_duel_out_of_field & (
        df_events1["team_id"] != df_events2["team_id"]
    )
    selector1_duel_won_air = selector1_duel_won  & (df_events1.type_secondary.apply(lambda x: x is not None and "aerial_duel" in x))
    selector1_duel_won_not_air = selector1_duel_won & (df_events1.type_secondary.apply(lambda x: x is not None and "ground_duel" in x))

    # Aggregate selectors
    selector_duel_won = selector0_duel_won | selector1_duel_won
    selector_duel_won_air = selector0_duel_won_air | selector1_duel_won_air
    selector_duel_won_not_air = selector0_duel_won_not_air | selector1_duel_won_not_air

    # Set types and subtypes to be passes instead of duels
    df_events.loc[selector_duel_won, "type_primary"] = 'pass'
    df_events.loc[selector_duel_won_air, "type_secondary"] = "head_pass"
    df_events.loc[selector_duel_won_not_air, "type_secondary"] = "pass"

    # set end location equal to ball out of field location
    df_events.loc[selector_duel_won, "accurate"] = False
    df_events.loc[selector_duel_won, "not_accurate"] = True
    df_events.loc[selector_duel_won, "end_x"] = 100 - df_events2.loc[selector_duel_won, "start_x"]
    df_events.loc[selector_duel_won, "end_y"] = 100 - df_events2.loc[selector_duel_won, "start_y"]

    # Define selector for ground attacking duels with take on
    selector_attacking_duel = df_events.type_secondary.apply(lambda x: x is not None and "offensive_duel" in x)
    selector_take_on = df_events.groundDuel_takeOn == True
    selector_att_duel_take_on = selector_attacking_duel & selector_take_on

    # Set take ons type to an empty string for now
    df_events.loc[selector_att_duel_take_on, "type_primary"] = None

    # Set sliding tackles type to an empty string for now
    df_events.loc[df_events.type_secondary.apply(lambda x: x is not None and "sliding_tackle" in x), "type_primary"] = None

    # Remove the remaining duels
    df_events = df_events[df_events["type_primary"] != "duel"]

    # Reset the index
    df_events = df_events.reset_index(drop=True)

    return df_events


def insert_interception_passes(df_events: pd.DataFrame) -> pd.DataFrame:
    """Insert interception actions before passes.

    This function converts passes (type_id 8) that are also interceptions
    (tag interception) in the Wyscout event data into two separate events,
    first an interception and then a pass.

    Parameters
    ----------
    df_events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.DataFrame
        Wyscout event dataframe in which passes that were also denoted as
        interceptions in the Wyscout notation are transformed into two events
    """
    df_events_interceptions = df_events[
        df_events.type_secondary.apply(lambda x: x is not None and "pass" in x) & (df_events.type_primary == 'interception')
    ].copy()

    if not df_events_interceptions.empty:
        df_events_interceptions["type_primary"] = "interception"
        df_events_interceptions["type_secondary"] = "interception"
        df_events_interceptions["id"] = df_events_interceptions["id"]
        df_events_interceptions[["end_x", "end_y"]] = df_events_interceptions[
            ["start_x", "start_y"]
        ]

        df_events = pd.concat([df_events_interceptions, df_events], ignore_index=True)
        df_events = df_events.sort_values(["period_id", "milliseconds"])
        df_events = df_events.reset_index(drop=True)

    return df_events


def add_offside_variable(df_events: pd.DataFrame) -> pd.DataFrame:
    """Attach offside events to the previous action.

    This function removes the offside events in the Wyscout event data and adds
    sets offside to 1 for the previous event (if this was a passing event)

    Parameters
    ----------
    df_events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.DataFrame
        Wyscout event dataframe with an extra column 'offside'
    """
    # Create a new column for the offside variable
    df_events["offside"] = 0

    # Shift events dataframe by one timestep
    df_events1 = df_events.shift(-1)

    # Select offside passes
    selector_offside = (df_events1["type_primary"] == "offisde") & (df_events["type_primary"] == "pass")

    # Set variable 'offside' to 1 for all offside passes
    df_events.loc[selector_offside, "offside"] = 1

    # Remove offside events
    df_events = df_events[df_events["type_primary"] != "offisde"]

    # Reset index
    df_events = df_events.reset_index(drop=True)

    return df_events


def convert_simulations(df_events: pd.DataFrame) -> pd.DataFrame:
    """Convert simulations to failed take-ons.

    Parameters
    ----------
    df_events : pd.DataFrame
        Wyscout event dataframe


    Returns
    -------
        pd.DataFrame
        Wyscout event dataframe in which simulation events are either
        transformed into a failed take-on
    """
    prev_events = df_events.shift(1)

    # Select simulations
    selector_simulation = df_events.infraction_type == "simulation_foul"

    # Select actions preceded by a failed take-on
    selector_previous_is_failed_take_on = ((prev_events.groundDuel_takeOn == True )
                                           & (prev_events.groundDuel_keptPossession != True))

    # Transform simulations not preceded by a failed take-on to a failed take-on
    df_events.loc[selector_simulation & ~selector_previous_is_failed_take_on, "type_primary"] = None
    df_events.loc[selector_simulation & ~selector_previous_is_failed_take_on, "type_secondary"] = None
    df_events.loc[selector_simulation & ~selector_previous_is_failed_take_on, "accurate"] = False
    df_events.loc[
        selector_simulation & ~selector_previous_is_failed_take_on, "not_accurate"
    ] = True
    # Set take_on_left or take_on_right to True
    df_events.loc[
        selector_simulation & ~selector_previous_is_failed_take_on, "groundDuel_takeOn"
    ] = True

    # Remove simulation events which are preceded by a failed take-on
    df_events = df_events[~(selector_simulation & selector_previous_is_failed_take_on)]

    # Reset index
    df_events = df_events.reset_index(drop=True)

    return df_events


def convert_touches(df_events: pd.DataFrame) -> pd.DataFrame:
    """Convert touch events to dribbles or passes.

    This function converts the Wyscout 'touch' event (sub_type_id 72) into either
    a dribble or a pass (accurate or not depending on receiver)

    Parameters
    ----------
    df_events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.DataFrame
        Wyscout event dataframe without any touch events
    """
    df_events1 = df_events.shift(-1)

    selector_touch = (df_events["type_primary"] == "touch")

    selector_same_player = df_events["player_id"] == df_events1["player_id"]
    selector_same_team = df_events["team_id"] == df_events1["team_id"]

    # selector_touch_same_player = selector_touch & selector_same_player
    selector_touch_same_team = selector_touch & ~selector_same_player & selector_same_team
    selector_touch_other = selector_touch & ~selector_same_player & ~selector_same_team

    same_x = abs(df_events["end_x"] - df_events1["start_x"]) < min_dribble_length
    same_y = abs(df_events["end_y"] - df_events1["start_y"]) < min_dribble_length
    same_loc = same_x & same_y
    df_events['same_loc'] = same_loc
    df_events['same_team'] = selector_same_team
    df_events['touch_other'] = selector_touch_other
    # return df_events
    df_events.loc[selector_touch_same_team & same_loc, "type_primary"] = "pass"
    df_events.loc[selector_touch_same_team & same_loc, "type_secondary"] = "pass"
    df_events.loc[selector_touch_same_team & same_loc, "accurate"] = True
    df_events.loc[selector_touch_same_team & same_loc, "not_accurate"] = False

    df_events.loc[selector_touch_other & same_loc, "type_primary"] = "pass"
    df_events.loc[selector_touch_other & same_loc, "type_secondary"] = "pass"
    df_events.loc[selector_touch_other & same_loc, "accurate"] = False
    df_events.loc[selector_touch_other & same_loc, "not_accurate"] = True

    return df_events


def create_df_actions(df_events: pd.DataFrame) -> pd.DataFrame:
    """Create the SciSports action dataframe.

    Parameters
    ----------
    df_events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.DataFrame
        SciSports action dataframe
    """
    df_events["time_seconds"] = df_events["milliseconds"] / 1000
    df_actions = df_events[
        [
            "game_id",
            "period_id",
            "time_seconds",
            "team_id",
            "player_id",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
        ]
    ].copy()
    df_actions["original_event_id"] = df_events["id"].astype(object)
    df_actions["bodypart_id"] = df_events.apply(determine_bodypart_id, axis=1)
    df_actions["type_id"] = df_events.apply(determine_type_id, axis=1)
    df_actions["result_id"] = df_events.apply(determine_result_id, axis=1)

    df_actions = remove_non_actions(df_actions)  # remove all non-actions left

    return df_actions


def determine_bodypart_id(event: pd.DataFrame) -> int:
    """Determint eht body part for each action.

    Parameters
    ----------
    event : pd.Series
        Wyscout event Series

    Returns
    -------
    int
        id of the body part used for the action
    """
    if (event.type_primary in ["save", "throw_in"]
        or event.infraction_type == "hand_foul"
        or (event.type_secondary != None and "hand_pass" in event.type_secondary)):
        body_part = "other"
    elif event.type_secondary != None and "head_pass" in event.type_secondary:
        body_part = "head"
    elif event.type_secondary != None and "head_shot" in event.type_secondary:
        body_part = "head/other"
    elif event.shot_bodyPart == "left_foot":
        body_part = "foot_left"
    elif event.shot_bodyPart == "right_foot":
        body_part = "foot_right"
    else:  # all other cases
        body_part = "foot"
    return spadlconfig.bodyparts.index(body_part)


def determine_type_id(event: pd.DataFrame) -> int:  # noqa: C901
    """Determine the type of each action.

    This function transforms the Wyscout events, sub_events and tags
    into the corresponding SciSports action type

    Parameters
    ----------
    event : pd.Series
        A series from the Wyscout event dataframe

    Returns
    -------
    int
        id of the action type
    """
    if event.type_primary == "fairplay":
        action_type = "non_action"
    elif event.type_primary == "own_goal":
        action_type = "bad_touch"
    elif event.type_primary == "pass":
        if "cross" in event.type_secondary:
            action_type = "cross"
        else:
            action_type = "pass"
    elif event.type_primary == "throw_in":
        action_type = "throw_in"
    elif event.type_primary == "corner":
        if event.pass_height == "high":
            action_type = "corner_crossed"
        else:
            action_type = "corner_short"
    elif event.type_primary == "free_kick":
        if event.type_secondary is not None and "free_kick_cross" in event.type_secondary:
            action_type = "freekick_crossed"
        else:
            action_type = "freekick_short"
    elif event.type_primary == "goal_kick":
        action_type = "goalkick"
    elif event.type_primary == "infraction" and (event.infraction_type not in ["protest_foul", "late_card_foul", "out_of_play_foul", "time_lost_foul"]):
        action_type = "foul"
    elif event.type_primary == "shot":
        action_type = "shot"
    elif event.type_primary == "penalty":
        action_type = "shot_penalty"
    elif event.type_secondary is not None and "free_kick_shot" in event.type_secondary:
        action_type = "shot_freekick"
    elif event.type_primary == "shot_against":
        action_type = "keeper_save"
    elif event.type_primary == "clearance":
        action_type = "clearance"
    elif event.type_primary == "touch" and event.not_accurate is True:  # Not accurate flag might cuase an issue
        action_type = "bad_touch"
    elif event.type_primary == "acceleration":
        action_type = "dribble"
    elif event.groundDuel_takeOn is True:
        action_type = "take_on"
    elif event.type_secondary is not None and "sliding_tackle" in event.type_secondary:
        action_type = "tackle"
    elif event.type_primary == "interception" and (event.type_secondary == "interception" or "recovery" in event.type_secondary):
        action_type = "interception"
    else:
        action_type = "non_action"
    return spadlconfig.actiontypes.index(action_type)


def determine_result_id(event: pd.DataFrame) -> int:  # noqa: C901
    """Determine the result of each event.

    Parameters
    ----------
    event : pd.Series
        Wyscout event Series

    Returns
    -------
    int
        result of the action
    """
    if event.type_primary == "offside":
        return 2
    if event.type_primary == "infraction":  # foul
        return 1
    if event.shot_isGoal is True:  # goal
        return 1
    if event.type_primary == "own_goal":  # own goal
        return 3
    if event.shot_isGoal is False:  # no goal, so 0
        return 0
    if event.pass_accurate is False:
        return 0
    if event["accurate"]:
        return 1
    if event["not_accurate"]:
        return 0
    # if (
    #     event.type_primary in ["interception", "clearance"]
    # ):  # interception or clearance always success
    #     return 1
    # if "save" in event.type_secondary:  # keeper save always success
    #     return 1
    # no idea, assume it was successful
    return 1


def remove_non_actions(df_actions: pd.DataFrame) -> pd.DataFrame:
    """Remove the remaining non_actions from the action dataframe.

    Parameters
    ----------
    df_actions : pd.DataFrame
        SciSports action dataframe

    Returns
    -------
    pd.DataFrame
        SciSports action dataframe without non-actions
    """
    df_actions = df_actions[df_actions["type_id"] != spadlconfig.actiontypes.index("non_action")]
    # remove remaining ball out of field, whistle and goalkeeper from line
    df_actions = df_actions.reset_index(drop=True)
    return df_actions


def fix_actions(df_actions: pd.DataFrame) -> pd.DataFrame:
    """Fix the generated actions.

    Parameters
    ----------
    df_actions : pd.DataFrame
        SPADL actions dataframe

    Returns
    -------
    pd.DataFrame
        SpADL actions dataframe with end coordinates for shots
    """
    df_actions["start_x"] = (df_actions["start_x"] * spadlconfig.field_length / 100).clip(
        0, spadlconfig.field_length
    )
    df_actions["start_y"] = (
        (100 - df_actions["start_y"])
        * spadlconfig.field_width
        / 100
        # y is from top to bottom in Wyscout
    ).clip(0, spadlconfig.field_width)
    df_actions["end_x"] = (df_actions["end_x"] * spadlconfig.field_length / 100).clip(
        0, spadlconfig.field_length
    )
    df_actions["end_y"] = (
        (100 - df_actions["end_y"])
        * spadlconfig.field_width
        / 100
        # y is from top to bottom in Wyscout
    ).clip(0, spadlconfig.field_width)
    df_actions = fix_goalkick_coordinates(df_actions)
    df_actions = adjust_goalkick_result(df_actions)
    df_actions = fix_foul_coordinates(df_actions)
    df_actions = fix_keeper_save_coordinates(df_actions)
    df_actions = remove_keeper_goal_actions(df_actions)
    df_actions.reset_index(drop=True, inplace=True)

    return df_actions


def fix_goalkick_coordinates(df_actions: pd.DataFrame) -> pd.DataFrame:
    """Fix goalkick coordinates.

    This function sets the goalkick start coordinates to (5,34)

    Parameters
    ----------
    df_actions : pd.DataFrame
        SciSports action dataframe with start coordinates for goalkicks in the
        corner of the pitch

    Returns
    -------
    pd.DataFrame
        SciSports action dataframe including start coordinates for goalkicks
    """
    goalkicks_idx = df_actions["type_id"] == spadlconfig.actiontypes.index("goalkick")
    df_actions.loc[goalkicks_idx, "start_x"] = 5.0
    df_actions.loc[goalkicks_idx, "start_y"] = 34.0

    return df_actions


def fix_foul_coordinates(df_actions: pd.DataFrame) -> pd.DataFrame:
    """Fix fould coordinates.

    This function sets foul end coordinates equal to the foul start coordinates

    Parameters
    ----------
    df_actions : pd.DataFrame
        SciSports action dataframe with no end coordinates for fouls

    Returns
    -------
    pd.DataFrame
        SciSports action dataframe including start coordinates for goalkicks
    """
    fouls_idx = df_actions["type_id"] == spadlconfig.actiontypes.index("foul")
    df_actions.loc[fouls_idx, "end_x"] = df_actions.loc[fouls_idx, "start_x"]
    df_actions.loc[fouls_idx, "end_y"] = df_actions.loc[fouls_idx, "start_y"]

    return df_actions


def fix_keeper_save_coordinates(df_actions: pd.DataFrame) -> pd.DataFrame:
    """Fix keeper save coordinates.

    This function sets keeper_save start coordinates equal to
    keeper_save end coordinates. It also inverts the shot coordinates to the own goal.

    Parameters
    ----------
    df_actions : pd.DataFrame
        SciSports action dataframe with start coordinates in the corner of the pitch

    Returns
    -------
    pd.DataFrame
        SciSports action dataframe with correct keeper_save coordinates
    """
    saves_idx = df_actions["type_id"] == spadlconfig.actiontypes.index("keeper_save")
    # invert the coordinates
    df_actions.loc[saves_idx, "end_x"] = (
        spadlconfig.field_length - df_actions.loc[saves_idx, "end_x"]
    )
    df_actions.loc[saves_idx, "end_y"] = (
        spadlconfig.field_width - df_actions.loc[saves_idx, "end_y"]
    )
    # set start coordinates equal to start coordinates
    df_actions.loc[saves_idx, "start_x"] = df_actions.loc[saves_idx, "end_x"]
    df_actions.loc[saves_idx, "start_y"] = df_actions.loc[saves_idx, "end_y"]

    return df_actions


def remove_keeper_goal_actions(df_actions: pd.DataFrame) -> pd.DataFrame:
    """Remove keeper goal-saving actions.

    This function removes keeper_save actions that appear directly after a goal

    Parameters
    ----------
    df_actions : pd.DataFrame
        SciSports action dataframe with keeper actions directly after a goal

    Returns
    -------
    pd.DataFrame
        SciSports action dataframe without keeper actions directly after a goal
    """
    prev_actions = df_actions.shift(1)
    same_phase = prev_actions.time_seconds + 10 > df_actions.time_seconds
    shot_goals = (prev_actions.type_id == spadlconfig.actiontypes.index("shot")) & (
        prev_actions.result_id == 1
    )
    penalty_goals = (prev_actions.type_id == spadlconfig.actiontypes.index("shot_penalty")) & (
        prev_actions.result_id == 1
    )
    freekick_goals = (prev_actions.type_id == spadlconfig.actiontypes.index("shot_freekick")) & (
        prev_actions.result_id == 1
    )
    goals = shot_goals | penalty_goals | freekick_goals
    keeper_save = df_actions["type_id"] == spadlconfig.actiontypes.index("keeper_save")
    goals_keepers_idx = same_phase & goals & keeper_save
    df_actions = df_actions.drop(df_actions.index[goals_keepers_idx])
    df_actions = df_actions.reset_index(drop=True)

    return df_actions


def adjust_goalkick_result(df_actions: pd.DataFrame) -> pd.DataFrame:
    """Adjust goalkick results.

    This function adjusts goalkick results depending on whether
    the next action is performed by the same team or not

    Parameters
    ----------
    df_actions : pd.DataFrame
        SciSports action dataframe with incorrect goalkick results

    Returns
    -------
    pd.DataFrame
        SciSports action dataframe with correct goalkick results
    """
    nex_actions = df_actions.shift(-1)
    goalkicks = df_actions["type_id"] == spadlconfig.actiontypes.index("goalkick")
    same_team = df_actions["team_id"] == nex_actions["team_id"]
    accurate = same_team & goalkicks
    not_accurate = ~same_team & goalkicks
    df_actions.loc[accurate, "result_id"] = 1
    df_actions.loc[not_accurate, "result_id"] = 0

    return df_actions


wyscout_periods = {"1H": 1, "2H": 2, "E1": 3, "E2": 4, "P": 5}
