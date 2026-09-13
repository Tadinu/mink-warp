"""Kinematic tasks."""

from .axis_match_task import AxisMatchTask as AxisMatchTask
from .com_task import ComTask as ComTask
from .damping_task import DampingTask as DampingTask
from .equality_constraint_task import EqualityConstraintTask as EqualityConstraintTask
from .frame_task import FrameTask as FrameTask
from .joint_limit_task import ConfigurationLimitTask as ConfigurationLimitTask
from .joint_limit_task import JointLimitTask as JointLimitTask
from .normal_match_task import NormalMatchTask as NormalMatchTask
from .pair_distance_task import PairDistanceTask as PairDistanceTask
from .point_match_task import PointAnchorTask as PointAnchorTask
from .point_match_task import PointMatchTask as PointMatchTask
from .posture_task import PostureTask as PostureTask
from .relative_frame_task import RelativeFrameTask as RelativeFrameTask
from .task import Task as Task
from .task import TargetedTask as TargetedTask
