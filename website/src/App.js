import './App.css';
import Navbar from "./Components/Navbar";
import Home from "./pages/Home";
import Survey from "./pages/Survey";
import {BrowserRouter as Router, Route, Routes} from "react-router-dom";

function App() {
  return (
  <div className="App">
    <Router>
      <Navbar/>
      <Routes>
        <Route path="/" element={<Home />}/>
        <Route path="/survey" element={<Survey />}/>
      </Routes>
    </Router>
  </div>);
}

export default App;
