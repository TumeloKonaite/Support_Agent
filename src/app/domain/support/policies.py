from src.app.domain.support.models import SupportKnowledge


def get_policy_entries(knowledge: SupportKnowledge) -> tuple[str, ...]:
    """Return policy entries from loaded knowledge content."""
    for section in knowledge.sections:
        if section.name.lower() == "policies":
            return section.entries

    return ()


def has_policy_content(knowledge: SupportKnowledge) -> bool:
    """Indicate whether policy knowledge is available for prompt construction."""
    return bool(get_policy_entries(knowledge))
