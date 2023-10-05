# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2022/12/13
# License: MIT License
"""Hooks design modified from pytorch.
"""
import weakref


class RemovableHandle:
    r"""
    A handle which provides the capability to remove a hook.
    Args:
        hooks_dict (dict): A dictionary of hooks, indexed by hook ``id``.
        extra_dict (dict): An additional dictionary whose keys will be deleted
            when the same keys are removed from ``hooks_dict``.
    """

    id: int
    next_id: int = 0

    def __init__(self, hooks_dict, extra_dict=None):
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

        self.extra_dict_ref = (
            weakref.ref(extra_dict) if extra_dict is not None else None
        )

    def remove(self):
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

        if self.extra_dict_ref is not None:
            extra_dict = self.extra_dict_ref()
            if extra_dict is not None and self.id in extra_dict:
                del extra_dict[self.id]

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.remove()
