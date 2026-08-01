from feast import Entity
from feast.value_type import ValueType

card_entity = Entity(
    name="card_id",
    join_keys=["card_id"],
    value_type=ValueType.STRING,
    description="Unique anonymized Payment Card / Account Identifier (card1)",
)

