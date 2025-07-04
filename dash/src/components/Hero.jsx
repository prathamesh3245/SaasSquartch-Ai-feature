// 

const Hero = () => {
  return (
    <section className="bg-white text-center py-24 px-6">
      <h1 className="text-red-500 text-8xl font-bold">
        Tailwind is working!
      </h1>
      <p className="text-gray-600 mb-8 text-lg max-w-xl mx-auto">
        If you see red text above, Tailwind is working.
      </p>

      <button
  onClick={() => window.location.href = 'http://localhost:5000'}
  className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg text-lg"
>
  Get Started
</button>
    </section>
  );
};

export default Hero;
