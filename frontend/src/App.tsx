import {
  Upload,
  Send,
  Bot,
} from 'lucide-react'

function App() {
  return (
    <main className="min-h-screen bg-gray-100 flex flex-col">

      {/* Header */}
      <header className="border-b bg-white px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center gap-3">
          
          <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center">
            <Bot className="text-white" size={22} />
          </div>

          <div>
            <h1 className="font-bold text-gray-800">
              YEAB Technologies
            </h1>

            <p className="text-sm text-gray-500">
              Employee Handbook Q&A
            </p>
          </div>

        </div>
      </header>


      {/* Chat Area */}
      <section className="flex-1 max-w-4xl w-full mx-auto px-6">

        {/* Welcome message */}
        <div className="flex flex-col items-center justify-center min-h-[70vh]">

          <div className="text-center mb-8">

            <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center mx-auto mb-5">
              <Bot
                size={32}
                className="text-blue-600"
              />
            </div>

            <h2 className="text-3xl font-bold text-gray-800 mb-3">
              Welcome to YEAB Technologies Chatbot
            </h2>

            <p className="text-gray-500">
              Ask me anything about the Employee Handbook
            </p>

          </div>


          {/* Question Input */}
          <div className="w-full max-w-2xl">

            <div className="bg-white border border-gray-300 rounded-2xl shadow-sm focus-within:ring-2 focus-within:ring-blue-500">

              <textarea
                className="w-full resize-none border-none outline-none p-5 rounded-2xl text-gray-700 placeholder-gray-400"
                rows={4}
                placeholder="Ask a question about the employee handbook..."
              />

              {/* Input buttons */}
              <div className="flex items-center justify-between px-4 pb-4">

                {/* Upload */}
                <button
                  className="flex items-center gap-2 px-3 py-2 rounded-lg text-gray-500 hover:bg-gray-100 hover:text-gray-700 transition"
                  title="Upload document"
                >
                  <Upload size={19} />
                  <span className="text-sm">
                    Upload
                  </span>
                </button>


                {/* Send */}
                <button
                  className="w-10 h-10 rounded-lg bg-blue-600 text-white flex items-center justify-center hover:bg-blue-700 transition"
                  title="Send question"
                >
                  <Send size={19} />
                </button>

              </div>

            </div>

            <p className="text-xs text-gray-400 text-center mt-3">
              Answers are generated from the YEAB Technologies Employee Handbook.
            </p>

          </div>

        </div>

      </section>

    </main>
  )
}

export default App