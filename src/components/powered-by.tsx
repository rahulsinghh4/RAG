const PoweredBy = () => {
  return (
    <p className="mt-4 text-xs md:text-sm text-zinc-50 text-center">
      This is a prototype RAG built for Physics Literature. <br /> Built using{" "}
      <a href="https://www.langchain.com/" target="_blank">
        LangChain
      </a>
      ,{" "}
      <a href="https://sdk.vercel.ai" target="_blank">
        Vercel AI SDK
      </a>{" "}
      and{" "}
      <a href="https://upstash.com" target="_blank">
        Upstash Vector
      </a>{" "}
      ・{" "}
      <a href="https://github.com/rahulsinghh4/RAG/" target="_blank">
        Source Code
      </a>
    </p>
  );
};

export default PoweredBy;
