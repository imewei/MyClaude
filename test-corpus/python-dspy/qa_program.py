"""DSPy RAG program compiled with MIPROv2 on a labelled trainset."""

import dspy
from dspy.evaluate import Evaluate


class AnswerWithContext(dspy.Signature):
    """Answer the question using only the provided context."""

    context = dspy.InputField(desc="relevant passages")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="short factual answer")


class RationaleWithCitation(dspy.Signature):
    """Produce a grounded rationale citing context passages by index."""

    context = dspy.InputField()
    question = dspy.InputField()
    rationale = dspy.OutputField(desc="numbered rationale with [i] citations")
    answer = dspy.OutputField()


class GroundedRAG(dspy.Module):
    """Multi-step RAG with retrieval, citation-grounded rationale, and answer."""

    def __init__(self, retriever, top_k: int = 5) -> None:
        super().__init__()
        self.retrieve = retriever
        self.top_k = top_k
        self.predict = dspy.ChainOfThought(RationaleWithCitation)

    def forward(self, question: str) -> dspy.Prediction:
        passages = self.retrieve(question, k=self.top_k)
        ctx = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages))
        return self.predict(context=ctx, question=question)


def em_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    return example.answer.strip().lower() in prediction.answer.strip().lower()


def compile_with_miprov2(
    program: dspy.Module,
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
    num_threads: int = 8,
) -> dspy.Module:
    optimizer = dspy.MIPROv2(
        metric=em_metric,
        auto="medium",
        num_threads=num_threads,
    )
    return optimizer.compile(program, trainset=trainset, valset=valset)


def bootstrap_fewshot(program: dspy.Module, trainset: list[dspy.Example]) -> dspy.Module:
    """Cheaper fallback: BootstrapFewShot as a smoke-test before MIPROv2."""
    return dspy.BootstrapFewShot(metric=em_metric).compile(program, trainset=trainset)


def evaluate_program(program: dspy.Module, devset: list[dspy.Example]) -> float:
    evaluator = Evaluate(devset=devset, metric=em_metric, num_threads=8)
    return evaluator(program)


def react_with_tools(glossary: dict[str, str]) -> dspy.ReAct:
    """ReAct module with a safe definition-lookup tool."""

    class DefinitionLookup(dspy.Tool):
        def __init__(self, terms: dict[str, str]) -> None:
            super().__init__(name="define", desc="look up a technical term")
            self.terms = terms

        def __call__(self, term: str) -> str:
            return self.terms.get(term.lower(), f"no definition for '{term}'")

    return dspy.ReAct(
        signature="question -> answer",
        tools=[DefinitionLookup(glossary)],
        max_iters=5,
    )


if __name__ == "__main__":
    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"), cache=False)
    program = GroundedRAG(retriever=my_retriever, top_k=5)  # noqa: F821
    compiled = compile_with_miprov2(program, trainset, valset)  # noqa: F821
    compiled.save("rag_compiled.json")
    print(evaluate_program(compiled, testset))  # noqa: F821
