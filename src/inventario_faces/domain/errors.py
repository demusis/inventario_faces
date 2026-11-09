from __future__ import annotations


class UserFacingError(Exception):
    """Condicao esperada de uso (nao e defeito do aplicativo).

    A mensagem ja e redigida para o usuario final: a interface a exibe integralmente,
    sem prefixo de tipo de excecao, e o log nao registra traceback.
    """


class NoEligibleFacesError(UserFacingError):
    """Um ou mais grupos da comparacao terminaram sem nenhuma face elegivel."""

    def __init__(self, message: str, *, empty_groups: tuple[str, ...]) -> None:
        super().__init__(message)
        self.empty_groups = empty_groups
