const Navbar = () => (
  <nav className="flex justify-between items-center p-6 shadow-md bg-white sticky top-0 z-50">
    <h7 className="text-sm font-bold text-gray-900 p-[10px]">SaasQuatch</h7>
    <div className="space-x-4">
      <button className="text-sm text-red-700 hover:text-black">Login</button>
      <button className="bg-black text-white px-5 py-2 rounded-full hover:bg-gray-800 transition">Sign Up</button>
    </div>
  </nav>
);

export default Navbar;
