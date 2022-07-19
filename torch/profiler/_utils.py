from collections import deque
from dataclasses import dataclass
from typing import Dict, List

from torch.profiler import DeviceType
from torch.autograd.profiler import profile
from torch.autograd import _KinetoEvent


@dataclass
class EventMetrics:
    duration_time_ns: int = 0
    self_time_ns: int = 0
    idle_time_ns: int = 0
    queue_depth: int = 0

    @property
    def fraction_idle_time(self):
        if self.duration_time_ns == 0:
            return 0.0
        return self.idle_time_ns / self.duration_time_ns


@dataclass
class Interval:
    start: int
    end: int
    queue_depth: int = 0


class EventKey:

    def __init__(self, event):
        self.event = event

    def __hash__(self):
        return hash(self.event.id)

    def __eq__(self, other):
        return self.event.id == other.event.id

    def __repr__(self):
        return f"<{self.event.name()} id={self.event.correlation_id}>"

    def intervals_overlap(self, intervals: List[Interval]):
        overlap_time = 0
        intervals = sorted(intervals, key=lambda x: x.start)
        for i, interval in enumerate(intervals):
            if i + 1 < len(intervals):
                assert interval.end <= intervals[
                    i + 1].start, "Intervals must be disjoint"
            overlap_start = max(self.event.start_time_ns, interval.start)
            overlap_end = min(self.event.end_time_ns, interval.end)

            if overlap_start < overlap_end:
                overlap_time += overlap_end - overlap_start
        return overlap_time


class BasicEvaluation:

    def __init__(self, prof: profile):
        self.profile = prof
        self.metrics: Dict[EventKey, EventMetrics] = {}
        self.compute_self_time()
        self.event_keys = sorted((e for e in self.metrics.keys()),
                                 key=lambda x: x.event.start_time_ns)
        self.events = [e.event for e in self.event_keys]
        self.cuda_events: List[_KinetoEvent] = []
        self.queue_depth_list = self.compute_queue_depth()
        self.compute_idle_time()

    def compute_self_time(self):
        '''
        Computes event's self time(total time - time in child ops).
        '''
        assert (self.profile.kineto_results is not None)
        stack = deque(self.profile.kineto_results.experimental_event_tree())

        # standard iterating dfs
        while stack:
            curr_event = stack.pop()
            self_time = curr_event.duration_time_ns
            for child_event in curr_event.children:
                self_time -= child_event.duration_time_ns
                stack.append(child_event)

            assert EventKey(
                curr_event
            ) not in self.metrics, f"Duplicate id: {curr_event.id}, {curr_event.name()}"
            self.metrics[EventKey(curr_event)] = EventMetrics(
                self_time_ns=self_time)
            self.metrics[EventKey(
                curr_event)].duration_time_ns = curr_event.duration_time_ns

    def compute_queue_depth(self):
        '''
        Computes queue_depth at each event. This will calculate the queue depth data for
        All the events in the tree.
        This will return a list of Interval of queue depth data of cuda launch and kernels.
        '''
        assert (self.profile.kineto_results is not None)
        cuda_event_list = self.profile.kineto_results.events()

        def is_cuda_launch_kernel(e):
            # TODO: find a better way to identify cudaLaunchKernel
            return e.name() == "cudaLaunchKernel"

        def is_cuda_kernel(e):
            # TODO: find a better way to identify CUDA Kernel
            return e.device_type() == DeviceType.CUDA and "mem" not in e.name(
            ).lower()

        cuda_launch_events = sorted(
            (e for e in cuda_event_list if is_cuda_launch_kernel(e)),
            key=lambda x: x.start_us())
        cuda_kernel_events = sorted(
            (e for e in cuda_event_list if is_cuda_kernel(e)),
            key=lambda x: x.start_us())

        self.cuda_events = sorted(cuda_launch_events + cuda_kernel_events,
                                  key=lambda x: x.start_us())

        kernel_mapping: Dict[_KinetoEvent, int] = {}
        last_mapped_kernel = 0
        for cuda_launch_event in cuda_launch_events:
            index = index_of_first_match(
                cuda_kernel_events,
                lambda x: x.linked_correlation_id(
                ) == cuda_launch_event.linked_correlation_id(),
                start=last_mapped_kernel)
            kernel_mapping[cuda_launch_event] = index
            last_mapped_kernel = index if index is not None else last_mapped_kernel

        current_kernel_index = 0
        spawned_kernel_index = -1

        all_events = cuda_launch_events + cuda_kernel_events + self.events

        def new_old_event_comparator(event):
            if hasattr(event, "start_us"):
                return event.start_us() * 1000
            if hasattr(event, "start_time_ns"):
                return event.start_time_ns
            raise Exception("Unknown Event Type")

        queue_depth_list: List[Interval] = []
        all_events.sort(key=new_old_event_comparator)
        for event in all_events:
            # Find latest cuda kernel event
            if hasattr(event, "start_us"):
                start_time = event.start_us() * 1000
                end_time = (event.start_us() + event.duration_us()) * 1000
                # Find current spawned cuda kernel event
                if event in kernel_mapping and kernel_mapping[
                        event] is not None:
                    spawned_kernel_index = kernel_mapping[event]
            elif hasattr(event, "start_time_ns"):
                start_time = event.start_time_ns  # type: ignore[attr-defined]
                end_time = event.end_time_ns  # type: ignore[attr-defined]

            while (current_kernel_index < len(cuda_kernel_events) and
                   (cuda_kernel_events[current_kernel_index].start_us()) * 1000
                   <= start_time):
                current_kernel_index += 1
            current_queue_depth = spawned_kernel_index - current_kernel_index + 1

            if hasattr(event, "start_us"):
                queue_depth_list.append(
                    Interval(start_time, end_time, current_queue_depth))
            elif hasattr(event, "start_time_ns"):
                self.metrics[EventKey(event)].queue_depth = current_queue_depth

        return queue_depth_list

    def compute_idle_time(self):
        '''
        Computes idle time of the profile.
        '''
        # Based on queue_depth_list, we can calculate idle time for all the events
        idle = False
        idle_start = 0
        idle_intervals: List[Interval] = []
        for data_point in self.queue_depth_list:
            if data_point.queue_depth == 0 and not idle:
                idle_start = data_point.end
                idle = True
            if data_point.queue_depth > 0 and idle:
                idle_intervals.append(Interval(idle_start, data_point.start))
                idle = False

        event_list = [e.event for e in self.metrics.keys()]
        for event in event_list:
            self.metrics[EventKey(event)].idle_time_ns = EventKey(
                event).intervals_overlap(idle_intervals)


def index_of_first_match(seq, predicate, start=0, end=None):
    if end is None or end >= len(seq):
        end = len(seq)
    for i in range(start, end):
        if predicate(seq[i]):
            return i
    return None
