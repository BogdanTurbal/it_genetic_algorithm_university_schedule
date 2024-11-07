import copy
import random
from typing import Dict, List, Optional, Tuple

import yaml


class TimeSlot:
    """Represents a specific time slot with a day and time."""

    def __init__(self, day: str, time: str):
        self.day: str = day
        self.time: str = time

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeSlot):
            return False
        return self.day == other.day and self.time == other.time

    def __str__(self) -> str:
        return f"{self.day}, {self.time}"

    def __repr__(self) -> str:
        return f"TimeSlot(day={self.day!r}, time={self.time!r})"

    def __hash__(self) -> int:
        return hash((self.day, self.time))


class Subject:
    """Represents a subject with a name and the number of hours it requires."""

    def __init__(self, name: str, hours: int):
        self.name: str = name
        self.hours: int = hours

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Subject):
            return False
        return self.name == other.name and self.hours == other.hours

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Subject(name={self.name!r}, hours={self.hours})"

    def __hash__(self) -> int:
        return hash((self.name, self.hours))


class Group:
    """Represents a student group with a name, capacity, and subjects."""

    def __init__(self, name: str, capacity: int, subject_names: List[str]):
        self.name: str = name
        self.capacity: int = capacity
        self.subject_names: List[str] = subject_names.copy()
        self.subjects: List[Subject] = []

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Group):
            return False
        return self.name == other.name and self.capacity == other.capacity

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Group(name={self.name!r}, capacity={self.capacity}, subject_names={self.subject_names!r})"

    def __hash__(self) -> int:
        return hash((self.name, self.capacity))


class Lecturer:
    """Represents a lecturer with a name and the subjects they can teach."""

    def __init__(self, name: str, can_teach_subjects_names: List[str]):
        self.name: str = name
        self.can_teach_subjects_names: List[str] = can_teach_subjects_names.copy()
        self.can_teach_subjects: List[Subject] = []

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Lecturer):
            return False
        return self.name == other.name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Lecturer(name={self.name!r}, can_teach_subjects_names={self.can_teach_subjects_names!r})"

    def __hash__(self) -> int:
        return hash(self.name)


class Hall:
    """Represents a hall with a name and capacity."""

    def __init__(self, name: str, capacity: int):
        self.name: str = str(name)  # Ensure name is always a string
        self.capacity: int = capacity

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hall):
            return False
        return self.name == other.name and self.capacity == other.capacity

    def __str__(self) -> str:
        return self.name  # Guaranteed to return a string

    def __repr__(self) -> str:
        return f"Hall(name={self.name!r}, capacity={self.capacity})"

    def __hash__(self) -> int:
        return hash((self.name, self.capacity))


class Slot:
    """Represents a scheduled class slot with group, subject, lecturer, hall, and time slot."""

    def __init__(
        self,
        group: Group,
        subject: Subject,
        lecturer: Optional[Lecturer] = None,
        hall: Optional[Hall] = None,
        time_slot: Optional[TimeSlot] = None,
    ):
        self.group: Group = group
        self.subject: Subject = subject
        self.lecturer: Optional[Lecturer] = lecturer
        self.hall: Optional[Hall] = hall
        self.time_slot: Optional[TimeSlot] = time_slot

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Slot):
            return False
        return (
            self.group == other.group
            and self.subject == other.subject
            and self.lecturer == other.lecturer
            and self.hall == other.hall
            and self.time_slot == other.time_slot
        )

    def __str__(self) -> str:
        return (
            f"Group: {self.group.name}, Subject: {self.subject.name}, "
            f"Lecturer: {self.lecturer}, Hall: {self.hall}, Time Slot: {self.time_slot}"
        )

    def __repr__(self) -> str:
        return (
            f"Slot(group={self.group!r}, subject={self.subject!r}, "
            f"lecturer={self.lecturer!r}, hall={self.hall!r}, time_slot={self.time_slot!r})"
        )

    def __hash__(self) -> int:
        return hash((self.group, self.subject, self.lecturer, self.hall, self.time_slot))

    def to_table_format(self) -> str:
        """Formats the slot information for table display."""
        return (
            f"{self.time_slot.day if self.time_slot else ''}\n"
            f"{self.time_slot.time if self.time_slot else ''}\n"
            f"{self.group.name if self.group else ''}\n"
            f"{self.subject.name if self.subject else ''}\n"
            f"{self.lecturer.name if self.lecturer else ''}\n"
            f"{self.hall.name if self.hall else ''}"
        )


class Schedule:
    """Represents a schedule containing a list of slots."""

    def __init__(self):
        self.grid: List[Slot] = []

    def mutate_slot(self, slot: Slot):
        """Placeholder for slot mutation logic."""
        pass  # Implement mutation logic if needed

    def _find_timeslot_conflicts(self, entity_attribute: str) -> Dict[str, List[TimeSlot]]:
        """General method to find time slot conflicts for a given entity attribute."""
        time_slots: Dict[str, List[TimeSlot]] = {}
        for slot in self.grid:
            entity = getattr(slot, entity_attribute)
            if entity is None or slot.time_slot is None:
                continue
            entity_name = entity.name
            time_slots.setdefault(entity_name, []).append(slot.time_slot)

        conflicts: Dict[str, List[TimeSlot]] = {}
        for entity_name, slots in time_slots.items():
            conflicts[entity_name] = [
                slot for slot in set(slots) if slots.count(slot) > 1
            ]
        return conflicts

    def is_valid(
        self,
    ) -> Tuple[
        bool, Dict[str, List[TimeSlot]], Dict[str, List[TimeSlot]], Dict[str, List[TimeSlot]]
    ]:
        """Check if the schedule is valid, i.e., no conflicts exist."""
        group_conflicts = self._find_timeslot_conflicts("group")
        hall_conflicts = self._find_timeslot_conflicts("hall")
        lecturer_conflicts = self._find_timeslot_conflicts("lecturer")

        is_valid_schedule = (
            not any(group_conflicts.values())
            and not any(hall_conflicts.values())
            and not any(lecturer_conflicts.values())
        )
        return is_valid_schedule, group_conflicts, hall_conflicts, lecturer_conflicts

    def crossover(self, other: "Schedule") -> "Schedule":
        """Perform crossover with another schedule to produce a child schedule."""
        child = Schedule()
        child.grid = self.grid.copy()

        for i in range(len(self.grid)):
            if random.random() > 0.5:
                child.grid[i] = copy.deepcopy(other.grid[i])

        return child

    def __str__(self) -> str:
        return "\n".join(str(slot) for slot in self.grid)

    def to_time_slot_oriented_view(self) -> Dict[str, List[Slot]]:
        """Return the schedule organized by time slots."""
        time_slots: Dict[str, List[Slot]] = {}
        for slot in self.grid:
            if slot.time_slot is None:
                continue
            key = f"{slot.time_slot.day} + {slot.time_slot.time}"
            time_slots.setdefault(key, []).append(slot)

        sorted_keys = sorted(time_slots.keys())
        return {k: time_slots[k] for k in sorted_keys}

    def to_groups_schedules(self) -> Dict[str, List[str]]:
        """Return the schedule organized by groups."""
        groups: Dict[str, List[Slot]] = {}
        for slot in self.grid:
            if slot.group is None:
                continue
            group_name = slot.group.name
            groups.setdefault(group_name, []).append(slot)

        group_tables: Dict[str, List[str]] = {}
        for group_name, slots in groups.items():
            sorted_slots = sorted(
                slots, key=lambda x: (x.time_slot.day, x.time_slot.time)
            )
            group_tables[group_name] = [slot.to_table_format() for slot in sorted_slots]

        return group_tables

    def _get_windows_cost(
        self, entity_attribute: str, time_slot_scores: Dict[TimeSlot, int]
    ) -> int:
        """Calculate the total cost for windows (gaps) for a given entity."""
        total_cost = 0
        entity_time_slots: Dict[str, List[TimeSlot]] = {}
        for slot in self.grid:
            entity = getattr(slot, entity_attribute)
            if entity is None or slot.time_slot is None:
                continue
            entity_name = entity.name
            entity_time_slots.setdefault(entity_name, []).append(slot.time_slot)

        for slots in entity_time_slots.values():
            sorted_time_slots = sorted(slots, key=lambda x: time_slot_scores[x])
            for i in range(len(sorted_time_slots) - 1):
                gap = (
                    time_slot_scores[sorted_time_slots[i + 1]]
                    - time_slot_scores[sorted_time_slots[i]]
                    - 1
                )
                total_cost += max(gap, 0)
        return total_cost

    def _get_time_slot_earliness_cost(
        self, time_slot_scores: Dict[TimeSlot, int]
    ) -> int:
        """Calculate the cost based on the earliness of time slots."""
        cost = 0
        for slot in self.grid:
            if slot.time_slot is not None:
                cost += time_slot_scores[slot.time_slot]
        return cost

    def _get_group_capacity_hall_capacity_fill_cost(self) -> float:
        """Calculate the cost based on how well the hall capacity matches the group size."""
        cost = 0.0
        for slot in self.grid:
            if slot.hall is None or slot.group is None:
                continue
            capacity_diff = slot.hall.capacity - slot.group.capacity
            cost += capacity_diff / slot.hall.capacity
        return cost


class ScheduleManager:
    """Manages the scheduling process using a genetic algorithm."""

    def __init__(self):
        self.time_slots: List[TimeSlot] = []
        self.subjects: List[Subject] = []
        self.groups: List[Group] = []
        self.lecturers: List[Lecturer] = []
        self.halls: List[Hall] = []
        self.time_slots_scores: Dict[TimeSlot, int] = {}

        self.group_windows_weight: float = 1.0
        self.lecturer_windows_weight: float = 1.0
        self.time_slot_earliness_weight: float = 0.5
        self.group_capacity_hall_capacity_fill_weight: float = 0.5

    def from_yaml(self, file_path: str) -> None:
        """Load scheduling data from a YAML file."""
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            self.time_slots = [TimeSlot(**ts) for ts in data.get("time_slots", [])]
            self.subjects = [
                Subject(**subject) for subject in data.get("subjects", [])
            ]
            self.groups = [Group(**group) for group in data.get("groups", [])]
            self.lecturers = [
                Lecturer(**lecturer) for lecturer in data.get("lecturers", [])
            ]
            self.halls = [Hall(**hall) for hall in data.get("halls", [])]

        for index, time_slot in enumerate(self.time_slots):
            self.time_slots_scores[time_slot] = index

    def create_empty_schedule(self) -> Schedule:
        """Create an empty schedule with slots for each group's subjects."""
        schedule = Schedule()
        for group in self.groups:
            for subject_name in group.subject_names:
                subject = next(
                    (s for s in self.subjects if s.name == subject_name), None
                )
                if subject is None:
                    continue
                for _ in range(subject.hours):
                    schedule.grid.append(Slot(group=group, subject=subject))
        return schedule

    def _get_available_time_slots(
        self, schedule: Schedule, hall: Hall, lecturer: Lecturer, group: Group
    ) -> List[TimeSlot]:
        """Get available time slots for given hall, lecturer, and group."""
        busy_time_slots = {
            slot.time_slot
            for slot in schedule.grid
            if slot.time_slot is not None
            and (slot.hall == hall or slot.lecturer == lecturer or slot.group == group)
        }
        return sorted(
            list(set(self.time_slots) - busy_time_slots),
            key=lambda x: (x.day, x.time),
        )

    def _get_available_lecturers(
        self, schedule: Schedule, time_slot: TimeSlot, subject: Subject
    ) -> List[Lecturer]:
        """Get available lecturers for a given time slot and subject."""
        busy_lecturers = {
            slot.lecturer
            for slot in schedule.grid
            if slot.time_slot == time_slot and slot.lecturer is not None
        }
        suitable_lecturers = [
            lecturer
            for lecturer in self.lecturers
            if subject.name in lecturer.can_teach_subjects_names
        ]
        return sorted(
            list(set(suitable_lecturers) - busy_lecturers), key=lambda x: x.name
        )

    def _get_available_halls(
        self, schedule: Schedule, time_slot: TimeSlot, group: Group
    ) -> List[Hall]:
        """Get available halls for a given time slot and group."""
        busy_halls = {
            slot.hall
            for slot in schedule.grid
            if slot.time_slot == time_slot and slot.hall is not None
        }
        suitable_halls = [
            hall for hall in self.halls if group.capacity <= hall.capacity
        ]
        return sorted(list(set(suitable_halls) - busy_halls), key=lambda x: x.name)

    def mutate_schedule(self, schedule: Schedule) -> Tuple[Schedule, bool]:
        """Mutate a schedule by changing one of its slots."""
        grid = schedule.grid
        slot_index = random.randint(0, len(grid) - 1)
        slot = grid[slot_index]
        new_slot = copy.deepcopy(slot)
        attribute_to_mutate = random.choice(["hall", "lecturer", "time_slot"])

        if attribute_to_mutate == "hall":
            available_halls = self._get_available_halls(
                schedule, slot.time_slot, slot.group
            )
            if not available_halls:
                return schedule, False
            new_slot.hall = random.choice(available_halls)
        elif attribute_to_mutate == "lecturer":
            available_lecturers = self._get_available_lecturers(
                schedule, slot.time_slot, slot.subject
            )
            if not available_lecturers:
                return schedule, False
            new_slot.lecturer = random.choice(available_lecturers)
        else:
            available_time_slots = self._get_available_time_slots(
                schedule, slot.hall, slot.lecturer, slot.group
            )
            if not available_time_slots:
                return schedule, False
            new_slot.time_slot = random.choice(available_time_slots)

        new_schedule = copy.deepcopy(schedule)
        new_schedule.grid[slot_index] = new_slot
        return new_schedule, True

    def init_random_schedule(self) -> Schedule:
        """Initialize a random schedule."""
        schedule = self.create_empty_schedule()
        attempts = 0
        max_attempts = 1000

        while attempts < max_attempts:
            schedule, mutated = self.mutate_schedule(schedule)
            if mutated:
                attempts += 1
            else:
                break

        # Fill None values
        for slot in schedule.grid:
            if slot.hall is None:
                available_halls = self._get_available_halls(
                    schedule, slot.time_slot, slot.group
                )
                if not available_halls:
                    return self.init_random_schedule()
                slot.hall = random.choice(available_halls)

            if slot.lecturer is None:
                available_lecturers = self._get_available_lecturers(
                    schedule, slot.time_slot, slot.subject
                )
                if not available_lecturers:
                    return self.init_random_schedule()
                slot.lecturer = random.choice(available_lecturers)

            if slot.time_slot is None:
                available_time_slots = self._get_available_time_slots(
                    schedule, slot.hall, slot.lecturer, slot.group
                )
                if not available_time_slots:
                    return self.init_random_schedule()
                slot.time_slot = random.choice(available_time_slots)

        return schedule

    def get_schedule_fitness(self, schedule: Schedule) -> float:
        """Calculate the fitness of a schedule."""
        group_windows_cost = self.group_windows_weight * schedule._get_windows_cost(
            "group", self.time_slots_scores
        )
        lecturer_windows_cost = (
            self.lecturer_windows_weight
            * schedule._get_windows_cost("lecturer", self.time_slots_scores)
        )
        time_slot_earliness_cost = (
            self.time_slot_earliness_weight
            * schedule._get_time_slot_earliness_cost(self.time_slots_scores)
        )
        capacity_fill_cost = (
            self.group_capacity_hall_capacity_fill_weight
            * schedule._get_group_capacity_hall_capacity_fill_cost()
        )

        total_cost = (
            group_windows_cost
            + lecturer_windows_cost
            + time_slot_earliness_cost
            + capacity_fill_cost
        )
        return 1 / (1 + total_cost)

    def tournament_selection(
        self, population: List[Schedule], tournament_size: int
    ) -> Schedule:
        """Select a schedule from the population using tournament selection."""
        selected = random.sample(population, tournament_size)
        return max(selected, key=lambda x: self.get_schedule_fitness(x))

    def genetic(self, population_size: int) -> Schedule:
        """Run the genetic algorithm to find the best schedule."""
        population = [self.init_random_schedule() for _ in range(population_size)]
        best_fitness = 0
        generations_without_improvement = 0
        max_generations = 100
        max_stagnation = 10

        while max_generations > 0 and generations_without_improvement < max_stagnation:
            new_population = []
            while len(new_population) < population_size:
                parent1 = self.tournament_selection(population, 10)
                parent2 = self.tournament_selection(population, 10)
                child = parent1.crossover(parent2)
                if random.random() < 0.5:
                    child, _ = self.mutate_schedule(child)
                if child.is_valid()[0]:
                    new_population.append(child)

            population = new_population
            current_best = max(population, key=lambda x: self.get_schedule_fitness(x))
            current_best_fitness = self.get_schedule_fitness(current_best)

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                generations_without_improvement = 0
                print(f"New best fitness: {best_fitness}")
            else:
                generations_without_improvement += 1

            max_generations -= 1

        return max(population, key=lambda x: self.get_schedule_fitness(x))
