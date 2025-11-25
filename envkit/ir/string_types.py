
from __future__ import annotations

import re
from typing import Any, ClassVar, cast, Union

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema
from typing_extensions import Self


# ----------------------------------------------------------------------
# 1. Metaclass – immutable regex
# ----------------------------------------------------------------------
class StringTypeMeta(type):
    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        dct: dict[str, Any],
        *,
        pattern: str | None = None,
    ) -> Self:
        if pattern is not None:
            dct["_PATTERN"] = re.compile(pattern)
        else:
            dct["_PATTERN"] = None

        cls = super().__new__(mcs, name, bases, dct)

        if pattern is not None and (not hasattr(cls, "_PATTERN") or cls._PATTERN is None):
            raise TypeError(
                f"Internal error: pattern was provided for {name!r} but "
                "_PATTERN was not set or is None."
            )

        return cast(Self, cls)

    def __setattr__(cls, name: str, value: Any) -> None:
        if name == "_PATTERN":
            raise AttributeError(f"{cls.__name__}._PATTERN is immutable")
        super().__setattr__(name, value)

    def __instancecheck__(cls, instance: Any) -> bool:
        if cls._PATTERN is None:
            return False
        return isinstance(instance, str) and bool(cls._PATTERN.match(instance))

    def __repr__(cls) -> str:
        if getattr(cls, "_PATTERN", None) is None:
            return f"<StringType {cls.__name__}: abstract>"
        return f"<StringType {cls.__name__}: {cls._PATTERN.pattern!r}>"


# ----------------------------------------------------------------------
# 2. Base class
# ----------------------------------------------------------------------
class StringType(metaclass=StringTypeMeta, pattern=None):
    """
    Base for regex-backed string *types*.

    - Do **not** instantiate: `SnakeId("x")` → TypeError
    - Subclasses must pass a `pattern=...` keyword argument.
    """

    __slots__ = ()

    _PATTERN: ClassVar[re.Pattern[str] | None]

    def __new__(cls, *args: Any, **kwargs: Any) -> "StringType":
        raise TypeError(f"{cls.__name__} is a type object, not a constructor")

    # ---------- Pydantic core schema ----------
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        def validate_str_value(value: str) -> str:
            if cls._PATTERN is None:
                raise NotImplementedError(
                    f"Cannot use abstract StringType {cls.__name__} "
                    "directly in a Pydantic model."
                )
            if not cls._PATTERN.match(value):
                raise ValueError(
                    f"{value!r} does not match pattern {cls._PATTERN.pattern!r}"
                )
            return value

        inner_str_schema = core_schema.str_schema()

        if cls._PATTERN is None:
            metadata = inner_str_schema.get("metadata") or {}
            metadata["description"] = "Abstract string type - not for direct use"
            inner_str_schema["metadata"] = metadata

        schema = core_schema.no_info_after_validator_function(
            validate_str_value,
            inner_str_schema,
            serialization=core_schema.to_string_ser_schema(),
        )
        return schema

    # ---------- Pydantic JSON schema ----------
    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        schema = handler(core_schema)

        if cls._PATTERN is None:
            schema.setdefault("description", "Abstract string type - not for direct use")
        else:
            schema.setdefault("pattern", cls._PATTERN.pattern)
            schema.setdefault(
                "description",
                f"String matching regex pattern `{cls._PATTERN.pattern}`",
            )

        return schema

    @classmethod
    def is_valid(cls, v: Any) -> bool:
        if cls._PATTERN is None:
            raise TypeError(f"Abstract StringType {cls.__name__} has no pattern")
        return isinstance(v, str) and bool(cls._PATTERN.match(v))


# ----------------------------------------------------------------------
# 3. Built-ins
# ----------------------------------------------------------------------
class SnakeId(StringType, pattern=r"^[a-z][a-z0-9_]*$"):
    """snake_case_id"""


class ScratchId(StringType, pattern=r"^_[a-z][a-z0-9_]*_$"):
    """_ephemeral_scratch_id_"""


class NameId(StringType, pattern=r"^[_A-Za-z][A-Za-z0-9_]*$"):
    """Relaxed identifier"""


class SymbolId(StringType, pattern=r"^[A-Z][A-Z0-9_]*$"):
    """Symbol identifier used in shapes, e.g. N_RED, FIELD_W."""


# FieldId is *conceptually* SnakeId ∪ ScratchId. At the type level we use a Union.
FieldId = Union[SnakeId, ScratchId]
