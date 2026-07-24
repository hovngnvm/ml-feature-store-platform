from feast import Entity

card_entity = Entity(
    name="card_id",
    join_keys=["card_id"],
    description="Unique anonymized Payment Card / Account Identifier (card1)",
)
